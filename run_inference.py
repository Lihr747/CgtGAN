import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import random
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD
import json

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                # 判断选出来的五个next token来自于哪个句子，更新维护的
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

def batched_generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    batch_size, prefix_len, embedding_size = embed.shape
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones((batch_size, beam_size), device=device)
    is_stopped = torch.zeros((batch_size, beam_size), device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.unsqueeze(1)
                generated = generated.repeat(1, beam_size, 1, 1).flatten(start_dim=0, end_dim=1) # 200 20 768
                # can be reshaped to batchsize * beamsize * prefix_len * 768
                # next_tokens, scores = next_tokens.view(-1, 1), scores.view(-1)
                if tokens is None:
                    tokens = next_tokens.unsqueeze(-1)
                else:
                    tokens = tokens.unsqueeze(1).repeat(1, beam_size, 1, 1).flatten(start_dim=0, end_dim=1)
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped.view(-1)] = -float(np.inf)
                logits[is_stopped.view(-1), 0] = 0
                logits = logits.view(batch_size, beam_size, -1)
                scores_sum = scores.unsqueeze(-1) + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths.unsqueeze(-1)
                scores_sum_average, next_tokens = scores_sum_average.view(batch_size, -1).topk(beam_size, -1)
                # 判断选出来的五个next token来自于哪个句子，更新维护的
                # scores_sum_average = scores_sum_average.view(-1)
                # next_tokens = next_tokens.view(-1)
                next_tokens_source = torch.div(next_tokens, scores_sum.shape[-1], rounding_mode='trunc')
                seq_lengths = seq_lengths.gather(dim=1, index=next_tokens_source)
                next_tokens = next_tokens % scores_sum.shape[-1]
                # next_tokens = next_tokens.unsqueeze(1)
                seq_len = tokens.shape[-1]
                tokens = tokens.gather(dim=1, index=next_tokens_source.unsqueeze(-1).expand(-1,-1,seq_len))
                tokens = torch.cat((tokens, next_tokens.unsqueeze(-1)), dim=-1)
                gen_len = generated.shape[1]
                gen_dim = generated.shape[2]
                generated = generated.view(batch_size, beam_size, gen_len, -1)
                generated = generated.gather(dim=1, index=next_tokens_source.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,gen_len,gen_dim))
                generated = generated.view(batch_size * beam_size, gen_len, -1)
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped.gather(dim=1, index = next_tokens_source)
            next_token_embed = model.gpt.transformer.wte(next_tokens.view(-1).squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index)
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_texts = []
    orders = scores.argsort(descending=True, dim=1)
    output_list = tokens.cpu().numpy()
    for b in range(batch_size):
        output = output_list[b]
        length = seq_lengths[b]
        order = orders[b]
        batch_output = []
        for o in order:
            batch_output.append(tokenizer.decode(output[o][:int(length[o])]))
        output_texts.append(batch_output)
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def batched_generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device
    if embed is not None:
        batch_size = embed.shape[0]
    else:
        batch_size = tokens.shape[0]
    stop_flag = torch.zeros((batch_size), device=device, dtype=torch.bool)
    seq_lengths = torch.zeros((batch_size), device=device)
    with torch.no_grad():

        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)

            generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):

            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0
            # remove index start index + 1, to avoid the first token > 0.8
            # indices_to_remove = sorted_indices[sorted_indices_to_remove]
            _, new_sorted_indices = torch.sort(sorted_indices)
            sorted_logits[sorted_indices_to_remove] = filter_value
            logits = sorted_logits.gather(1, new_sorted_indices)
            next_token = torch.argmax(logits, -1).unsqueeze(1)
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            seq_lengths[~stop_flag] += 1
            stop_flag = stop_flag | (stop_token_index == next_token).squeeze()
            if stop_flag.sum() == batch_size:
                break
        output_list = list(tokens.cpu().numpy())
        batch_output = []
        for b in range(batch_size):
            output = output_list[b]
            length = seq_lengths[b]
            batch_output.append(tokenizer.decode(output[:int(length)]))
    return batch_output

def json_writer(values, f):
    res = []
    for i, g in values:
        res.append({"image_id":str(int(i)), "caption":g})
    json_str = json.dumps(res, indent=4)
    f.write(json_str)

def evalation(model, epoch_id, output_dir, gt_file, tokenizer, data_loader, device, use_beam_search = False, beam = 5, mode='val'):
    def _generate():
        with torch.no_grad():
            for idx, (tokens, mask, prefix, img_id) in enumerate(tqdm(data_loader)):
                prefix = prefix.to(device)
                prefix_embed = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)
                if use_beam_search:
                    generated_text_prefix = batched_generate_beam(model, tokenizer, beam_size=beam, embed=prefix_embed)
                    for i, g in zip(img_id, generated_text_prefix):
                        yield i, g[0]
                else:
                    generated_text_prefix = batched_generate2(model, tokenizer, embed=prefix_embed)
                    for i, g in zip(img_id, generated_text_prefix):
                        yield i, g
    f = open(os.path.join(output_dir, "epoch{}_{}.json".format(str(epoch_id), mode)), 'w')
    json_writer(_generate(), f)
    coco = COCO(gt_file)
    cocoRes = coco.loadRes(os.path.join(output_dir, "epoch{}_{}.json".format(str(epoch_id), mode)))
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    print(result)
    return result