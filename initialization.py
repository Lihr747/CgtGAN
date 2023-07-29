import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_constant_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union, overload
from run_inference import evalation
import random
import numpy as np
from utils.misc import mkdir, set_seed
from models import ClipCocoDatasetCaptionWise, ClipCaptionModel, MLP
from models import ClipCaptionModel
from models import MLP
import os.path as op
import time
from utils.distributional_utils import json_writer
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def save_model(model, args):
    checkpoint_dir = args.output_dir
    mkdir(checkpoint_dir)
    # model.module is the model self of DDP distributional setting
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(
            model_to_save.state_dict(),
            op.join(checkpoint_dir, "model.pt"),
            )
    print("Save checkpoint to {}".format(checkpoint_dir))
    return checkpoint_dir


def train(dataset, model: ClipCaptionModel, args):
    lr = args.learning_rate
    warmup_steps = args.warmup_steps
    device = torch.device('cuda')
    batch_size = args.batch_size
    output_dir = args.output_dir
    num_epochs = args.num_train_epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    sys.stdout.flush()

    for epoch in range(num_epochs):
        print("train epoch {}".format(epoch + 1))
        progress = tqdm(total=len(dataloader))
        for step, (tokens, mask, prefix) in enumerate(dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs_sentence = model(tokens, prefix, mask)
            logits_sentence = outputs_sentence.logits[:, args.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits_sentence.reshape(-1, logits_sentence.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="Linear warmup.")
    parser.add_argument("--num_train_epochs", default=1, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization.")
    parser.add_argument('--batch_size', type=int, default=16, help="The prefix seq length")
    parser.add_argument('--mapping_type', type=str, default="mlp", help="mapper type: mlp or transformer")
    parser.add_argument('--prefix_length', type=int, default=10, help="The prefix seq length")
    parser.add_argument('--prefix_length_clip', type=int, default=10,
                        help="When using transformer, project to this length by linear")
    parser.add_argument('--num_layers', type=int, default=4, help="transformer layer number")
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--clip_model_type', default="ViT-L/14")
    # adersial training setting
    args = parser.parse_args()
    prefix_length = args.prefix_length
    set_seed(args.seed, 1)

    if args.clip_model_type == "ViT-B/32":
        prefix_dim = 512
    elif args.clip_model_type == "ViT-L/14":
        prefix_dim = 768
    elif args.clip_model_type == "RN50x4":
        prefix_dim = 640
    else:
        raise RuntimeError('CLIP model type error')
    
    print("Training/evaluation parameters: ", args)

    model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                              num_layers=args.num_layers, mapping_type=args.mapping_type)
    print("Train both prefix and GPT")
    sys.stdout.flush()

    dataset = ClipCocoDatasetCaptionWise(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    model = train(dataset, model, args)
    save_model(model, args)

if __name__ == '__main__':
    main()
