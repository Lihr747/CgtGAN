import argparse
from cProfile import label
from tkinter import image_names
from typing import OrderedDict, Tuple, Optional, Union, overload 
import torch
from torch.nn import functional as nnf
import os
import os.path as op
from utils.logger import setup_logger
from utils.misc import mkdir, set_seed
from models import ClipCaptionModel, MLP, Mapping_Network
from models import ClipCocoDatasetImageWise, ClipCocoDatasetCaptionWise, TextFeatures, PairsFeatures
from models import ScstRewardCLIPCriterion
from models import RobertaDiscriminator
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import json
from tqdm import tqdm
from utils.distributional_utils import json_writer, concat_json_files, delete_json_files
import time
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from utils.distributional_utils import get_rank, is_main_process, synchronize, ensure_init_process_group, get_world_size
from enum import Enum
from run_inference import evalation
import gc
import random
import numpy as np

# make dataloader
def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, mode, is_distributed=True, is_train=True):
    if mode == "train":
        shuffle = True
        # Main Generator dataloader
        image_dataset = PairsFeatures(args.data_train)

        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(image_dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs  

        image_sampler = make_data_sampler(image_dataset, shuffle, is_distributed)

        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, num_workers=args.num_workers, sampler=image_sampler,
            batch_size=images_per_gpu,
            pin_memory=True, drop_last=True if mode == 'train' else False,
        )

        # Main Discriminator dataloader                                 
        text_dataset = TextFeatures(args.train_sample_n, args.text_corpus,
                                      args.max_gen_length, args.discriminator_type)

        text_sampler = make_data_sampler(text_dataset, shuffle, is_distributed)

        text_dataloader = torch.utils.data.DataLoader(
            text_dataset, num_workers=args.num_workers, sampler=text_sampler,
            batch_size=images_per_gpu,
            pin_memory=True, drop_last=True if mode == 'train' else False,
        )

        # log info
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
        logger.info("load without concate")
        return image_dataloader, text_dataloader

    else:
        # TODO: tensorize and tokenize caption
        if mode == "val":
            dataset = ClipCocoDatasetImageWise(args.gt_val, args.data_val, args.prefix_length,
                                               normalize_prefix=args.normalize_prefix, mode="val")
        elif mode == "test":
            dataset = ClipCocoDatasetImageWise(args.gt_test, args.data_test, args.prefix_length,
                                               normalize_prefix=args.normalize_prefix, mode="test")
        else:
            raise ValueError

        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        data_loader = torch.utils.data.DataLoader(
            dataset, num_workers=args.num_workers, sampler=sampler,
            batch_size=images_per_gpu,
            pin_memory=True, drop_last=True if mode == 'train' else False,
        )
        return data_loader
    
# do train    

def generate_with_baseline(args, img_feats, model, tokenizer):
    inputs = {
        'img_feats': img_feats,
        'do_sample': False,
        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": 1,
        "num_keep_best": 1,
        "eos_token": args.eos_token,
        "is_decode": True
    }

    def _ids_to_captions(all_ids):
        captions = []
        for ids in all_ids:
            c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            captions.append(c)
        return captions

    if args.sc_baseline_type == 'greedy':
        model.eval()
        with torch.no_grad():
            greedy_res_raw, _ = model(**inputs)
            greedy_res_raw.squeeze_(1)  # batch_size * max_len
        greedy_res = _ids_to_captions(greedy_res_raw)
    else:
        greedy_res = None

    model.train()
    inputs['do_sample'] = True
    inputs['num_return_sequences'] = args.train_sample_n
    sample_res_raw, sample_logprobs = model(**inputs)
    sample_res_raw.squeeze_(1)
    sample_logprobs.squeeze_(1)
    assert sample_logprobs.requires_grad == True
    assert sample_res_raw.requires_grad == False
    sample_res = _ids_to_captions(sample_res_raw)

    # gt_res = [train_dataloader.dataset.get_captions_by_key(k) for k in img_keys]
    return greedy_res, sample_res, sample_logprobs


def save_checkpoint(model, args, save_type, epoch=0, step=0, num_trial=10):
    if save_type == 'checkpoint':  
        checkpoint_dir = op.join(args.output_dir, 'checkpoint')
    elif save_type == 'best_model':
        checkpoint_dir = op.join(args.output_dir, 'best')
    
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    # model.module is the model self of DDP distributional setting
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(
                model_to_save.state_dict(),
                op.join(checkpoint_dir, "model.pt"),
            )
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def train(image_dataloader, text_dataloader, val_dataloader, generator, discriminator, args):
    if args.distributed:
        generator = torch.nn.parallel.DistributedDataParallel(
            generator, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        discriminator = torch.nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    t_total = len(image_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs 

    no_decay = ['bias', 'LayerNorm.weight']

    gp_G = [
        {'params': [p for n, p in generator.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in generator.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    gp_D = [
        {'params': [p for n, p in discriminator.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in discriminator.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_G = AdamW(gp_G, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer_D = AdamW(gp_D, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.scheduler == "constant":
        scheduler_G = get_constant_schedule_with_warmup(
            optimizer_G, num_warmup_steps=args.warmup_steps)
        scheduler_D = get_constant_schedule_with_warmup(
            optimizer_D, num_warmup_steps=0)
        
    elif args.scheduler == "linear":
        scheduler_G = get_linear_schedule_with_warmup(
            optimizer_G, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        scheduler_D = get_linear_schedule_with_warmup(
            optimizer_D, num_warmup_steps=0, num_training_steps=t_total)
        
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
    
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    logger.info("***** Running training *****")
    logger.info("  Epoch number = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                args.per_gpu_train_batch_size * get_world_size() * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    scst_criterion_CLIP = ScstRewardCLIPCriterion(
        args=args,
        baseline_type=args.sc_baseline_type,
    )

    logger.info("SCST training... ")

    global_step = 0
    best_score = 0.0

    cos_list = []
    l1_list = []
    clip_list = []

    epoch_steps = len(image_dataloader) // args.gradient_accumulation_steps 

    if args.distributed:
        tokenizer = generator.module.tokenizer
        pad_and_token = discriminator.module.pad_and_token
        PG_loss = discriminator.module.PG_loss
        get_critic_score = discriminator.module.get_score
    else:
        tokenizer = generator.tokenizer
        pad_and_token = discriminator.pad_and_token
        PG_loss = discriminator.PG_loss
        get_critic_score = discriminator.get_score
    

    for epoch in range(int(args.num_train_epochs)):
        if args.distributed:
            image_dataloader.sampler.set_epoch(epoch)
            text_dataloader.sampler.set_epoch(epoch)

        it_text = iter(text_dataloader)

        cos_total = 0.
        l1_total = 0.
        clip_total = 0.

        for step, (image_feats, text_feats) in enumerate(image_dataloader):
            # Training Main Network
            ## process data
            real_tokens, real_masks = it_text.next()
            real_tokens = real_tokens.to(args.device)
            real_masks = real_masks.to(args.device)
            bsz, sample_times, seq_len = real_tokens.shape
            real_tokens = real_tokens.view((bsz * sample_times, seq_len))
            real_masks = real_masks.view((bsz * sample_times, seq_len))
            image_feats, text_feats = image_feats.to(args.device), text_feats.to(args.device)
            if args.normalize_prefix:
                image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

            ## training Discriminator
            greedy_res, sample_res, sample_logprobs = generate_with_baseline(args, image_feats, generator, tokenizer)

            sample_res_token, sample_res_mask = pad_and_token(sample_res)
            text_real_labels = torch.ones((bsz * sample_times))
            text_fake_labels = torch.zeros((sample_res_token.shape[0]))
            D_loss = discriminator(torch.cat((real_tokens, sample_res_token.detach())),
                             torch.cat((real_masks, sample_res_mask.detach())), training=True,
                             labels=torch.cat((text_real_labels, text_fake_labels)))

            ## training Generator
            reward_loss = scst_criterion_CLIP(image_feats, text_feats, greedy_res, sample_res, sample_logprobs)
            fd_loss = PG_loss(greedy_res, sample_res, sample_logprobs)
            ## compute scores for main network
            l1_score, cos_score, clip_score = scst_criterion_CLIP.get_score()
            cos_total += cos_score
            l1_total += l1_score 
            clip_total += clip_score
            critic_score = get_critic_score()

            # loss backward and logging
            if args.gradient_accumulation_steps > 1:
                D_loss = D_loss / args.gradient_accumulation_steps
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler_D.step()
                discriminator.zero_grad()

            if global_step > args.gan_training_steps:
                reward_weight = min((global_step - args.gan_training_steps) / args.gan_warm_steps, 1.0) * args.reward_weight 
            else:
                reward_weight = 0.0                                 
            G_loss = reward_weight * reward_loss + (1.0 - reward_weight) * fd_loss    

            if args.gradient_accumulation_steps > 1:
                G_loss = G_loss / args.gradient_accumulation_steps
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm)       
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer_G.step()
                scheduler_G.step()
                generator.zero_grad()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}\n" \
                                "G loss: {:.4f}, D loss: {:.4f}".format(epoch + 1, global_step,
                                                                           G_loss, D_loss))       
                    logger.info("Main Network Training: \n" \
                                "cosine score: {:.4f}, l1 score: {:.4f}, clip score: {:.4f}, critic score: {:.4f}\n" \
                                "reward loss: {:.4f}, fD loss: {:.4f}\n".format(cos_score, l1_score, clip_score, critic_score,
                                                                                 reward_loss, fd_loss))
                    
        checkpoint_dir = save_checkpoint(generator, args, save_type='checkpoint')
        synchronize()
        # evaluation
        if args.evaluate_during_training:
            logger.info("Epoch %d training finish. Performing evaluation" % (epoch + 1))
            res = evaluate(args, global_step, val_dataloader, generator, tokenizer,
                    checkpoint_dir)
            if is_main_process():
                logger.info('evaluation result: {}\n'.format(str(res)))
                cider_score = res['CIDEr']
                if cider_score > best_score:
                    best_score = cider_score
                    logger.info("epoch: {}, best cider: {}".format(epoch + 1, best_score))
                    best_dir = save_checkpoint(generator, args, save_type='best_model', epoch=epoch + 1, step=global_step)                                
            synchronize()

        cos_total /= epoch_steps
        l1_total /= epoch_steps
        clip_total /= epoch_steps
        cos_list.append(cos_total)
        l1_list.append(l1_total)
        clip_list.append(clip_total)

        logger.info("average cos score: {} \n " \
                    "average l1 score: {} \n " \
                    "average clip score: {}".format(str(cos_list), str(l1_list), str(clip_list)))
    return 0

# do test/val


def test(args, test_dataloader, model, tokenizer, predict_file):
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(), 
                world_size) + op.splitext(predict_file)[1]
    model.eval()
    inputs_param = {'is_decode': True,
        'do_sample': False,
        # hyperparameters of beam search
        'eos_token' : args.eos_token,
        'max_length': args.max_gen_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_keep_best": args.num_keep_best,
    }
    def _generate():
        time_meter = 0
        with torch.no_grad():
            for step, (img_id, batch) in enumerate(test_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"img_feats":batch[2]}
                inputs.update(inputs_param)
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_id, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, res[0]['caption']
        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step+1)))
    cache_file = open(cache_file, 'w')
    json_writer(_generate(), cache_file)
    if world_size > 1:
        torch.distributed.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_json_files(cache_files, predict_file)
        delete_json_files(cache_files)
    if world_size > 1:
        torch.distributed.barrier()

def get_predict_file(output_dir, mode, iteration, args):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
    # data = op.basename(op.join(args.data_dir, '')[:-1])
    # cc.append(data)
    cc.append('max{}'.format(args.max_gen_length))
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    return op.join(output_dir, '{}_{}_{}.json'.format('.'.join(cc), iteration, mode))

def evaluate_on_coco_caption(predict_file, caption_file):
    coco = COCO(caption_file)
    cocoRes = coco.loadRes(predict_file)
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    return result

def evaluate(args, iteration, val_dataloader, model, tokenizer, output_dir):
    result = None
    mode = val_dataloader.dataset.mode
    predict_file = get_predict_file(output_dir, mode, iteration, args)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        torch.distributed.barrier()
    if is_main_process():
        caption_file = val_dataloader.dataset.gt_file
        result = evaluate_on_coco_caption(predict_file, caption_file)
    if get_world_size() > 1:
        torch.distributed.barrier()
    return result


def main():
    parser = argparse.ArgumentParser()

    # directory settings
    parser.add_argument('--data_train', type=str, required=True, help="path to pair embedding file")
    parser.add_argument('--text_corpus', type=str, required=True, help="path to text corpus file")
    parser.add_argument('--data_val', type=str, required=True, help="path to val embedding file")
    parser.add_argument('--data_test', type=str, required=True, help="path to test embedding file")
    parser.add_argument('--gt_val', type=str, required=True, help="path to val json file")
    parser.add_argument('--gt_test', type=str, required=True, help="path to test json file")
    parser.add_argument('--output_dir', type=str, default='./output', help="path to output")
    # operation settings
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_infer", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    # global setting
    parser.add_argument("--generator_init", type=str, default='', help="Generator checkpoint.")
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training.")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    # model setting
    parser.add_argument('--clip_model_type', default="ViT-L/14", help="ViT-L/14 or RN50x4 or ViT-B/32")
    parser.add_argument('--mapping_type', type=str, default="mlp", help="mapper type: mlp or transformer")
    parser.add_argument('--prefix_length', type=int, default=10, help="The prefix seq length")
    parser.add_argument('--prefix_length_clip', type=int, default=10,
                        help="When using transformer, project to this length by linear")
    parser.add_argument('--num_layers', type=int, default=4, help="transformer layer number")
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--discriminator_type', default='roberta-base')
    # training settings
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial lr for G and D")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=150, type=int, help="lr linear warmup.")
    parser.add_argument("--scheduler", default='constant', type=str, help="constant or linear")
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--logging_steps', type=int, default=25, help="Log every X steps.")
    parser.add_argument("--evaluate_during_training", type=bool, default=True,
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--test_after_training", type=bool, default=True,
                        help="Test best model after trainig.")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--train_sample_n', type=int, default=5,
                        help="number of sampled captions for sc training")
    # param settings
    parser.add_argument("--clip_weight", default=0., type=float, help="weight balancing clip reward and agg reward")
    parser.add_argument("--cos_weight", default=0.5, type=float, help="weight balancing cos and l1 in agg reward")
    parser.add_argument("--reward_weight", default=0.5, type=float, help="weight balancing reward and fD")
    parser.add_argument("--gan_training_steps", type=int, default=150, help="GAN training steps")
    parser.add_argument("--gan_warm_steps", type=int, default=2350, help="GAN warmup steps")
    # generation settings
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=20,
                        help="max length of generated sentences")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    parser.add_argument('--eos_token', default='.')
    # eval settings
    parser.add_argument('--eval_mode', default="test", help="test dataset: test or val")
    args = parser.parse_args()
    global logger

    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    output_dir = args.output_dir

    logger = setup_logger("Trainer", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)

    prefix_length = args.prefix_length

    if args.clip_model_type == "ViT-B/32":
        prefix_dim = 512
    elif args.clip_model_type == "ViT-L/14":
        prefix_dim = 768
    elif args.clip_model_type == "RN50x4":
        prefix_dim = 640
    else:
        raise RuntimeError('CLIP model type error')

    # create model
    if args.do_train:
        # create main generator
        generator_checkpoint = args.generator_init
        generator = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type)
        
        total = sum([param.nelement() for param in generator.parameters()])
        print("Number of parameter: %.2fM" % (total))
        
        if generator_checkpoint != "":
            logger.info(f"loading generator from {generator_checkpoint}")
            state_dict = torch.load(generator_checkpoint, map_location=torch.device('cpu'))
            generator.load_state_dict(state_dict)

        # create main discriminator
        discriminator = RobertaDiscriminator(args, args.sc_baseline_type, args.discriminator_type)
        discriminator.to(args.device)
        
    else:
        generator_checkpoint = args.generator_init 
        generator = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type)

        if generator_checkpoint != "":
            logger.info(f"loading generator from {generator_checkpoint}")
            state_dict = torch.load(generator_checkpoint, map_location=torch.device('cpu'))
            generator.load_state_dict(state_dict)

    generator.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # create dataloader and train
    if args.do_train:
        # dataloader for text_generatior, text_discriminatior, image_generatior, image_discriminatior
        image_dataloader, text_dataloader = \
        make_data_loader(args, "train", is_distributed=args.distributed, is_train=True)
        val_dataloader = None
        if args.evaluate_during_training:
            val_dataloader = make_data_loader(args, "val", is_distributed=args.distributed, is_train=False)

        train(image_dataloader, text_dataloader, val_dataloader, generator, discriminator, args)

        # test best model after training
        if args.test_after_training:
            del val_dataloader, image_dataloader, text_dataloader, discriminator
            gc.collect()
            torch.cuda.empty_cache()
            test_dataloader = make_data_loader(args, "test", is_distributed=args.distributed, is_train=False)
            logger.info("Testing best checkpoint")
            best_checkpoint = op.join(args.output_dir, 'best/model.pt')
            logger.info(f"loading generator from {best_checkpoint}")
            state_dict = torch.load(best_checkpoint, map_location=torch.device('cpu'))
            generator.load_state_dict(state_dict)
            generator.to(args.device)
            result = evaluate(args, "testtime", test_dataloader, generator, generator.tokenizer,
                                    args.output_dir)
            logger.info("Evaluation results : {}".format(str(result)))

    # inference and evaluation
    elif args.do_infer or args.do_eval:
        logger.info("Evaluate on dataset: " + args.eval_mode)
        test_dataloader = make_data_loader(args, args.eval_mode, is_distributed=args.distributed, is_train=False)

        if not args.do_eval:
            predict_file = get_predict_file(args.output_dir, args.eval_mode, "testtime", args)
            test(args, test_dataloader, generator, generator.tokenizer, predict_file)
            logger.info("Prediction results saved to: {}".format(predict_file))
        else:
            result = evaluate(args, "testtime", test_dataloader, generator, generator.tokenizer,
                              args.output_dir)
            logger.info("Evaluation results : {}".format(str(result)))


if __name__ == "__main__":
    main()
