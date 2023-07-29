from lib2to3.pgen2.tokenize import generate_tokens
import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RobertaTokenizer, RobertaModel
from tqdm import tqdm
import os
import pickle
import sys
from typing import Tuple, Optional, Union
import numpy as np
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import clip
from utils.inference import BeamHypotheses, top_k_top_p_filtering
from utils.distributional_utils import AllGather, is_main_process, synchronize
all_gather = AllGather.apply
from torch.autograd import Variable, grad


########## Datasets
class ClipCocoDatasetCaptionWise(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        i = item % len(self.captions_tokens)
        tokens, mask = self.pad_tokens(i)
        prefix = self.prefixes[self.caption2embedding[i]]
        prefix = prefix.float()
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)
        return (tokens, mask, prefix)

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False, max_seq_len = 20):
        self.max_seq_len = max_seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Caption Data size for generation is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        print("Clip embedding dim is %0d" % self.prefixes.shape[1])
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self.captions_tokens))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class ClipCocoDatasetImageWise(Dataset):
    
    def __len__(self) -> int:
        return len(self.image_ids)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item][0:5]
        # Attention: Some GT reference have 6 seqs, however, we only take 5 of them to keep the same tensor size
        # Eval scores is generated using GT file not the tokens, the tokens only for debug
        assert len(tokens) == 5
        masks = []
        for i in range(len(tokens)):
            padding = self.max_seq_len - tokens[i].shape[0]
            if padding > 0:
                tokens[i] = torch.cat((tokens[i], torch.zeros(padding, dtype=torch.int64) - 1))
                self.captions_tokens[item][i] = tokens[i]
            elif padding < 0:
                tokens[i] = tokens[i][:self.max_seq_len]
                self.captions_tokens[item][i] = tokens[i]
            mask = tokens[i].ge(0)  # mask is zero where we out of sequence
            tokens[i][~mask] = 0
            mask = mask.float()
            mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
            masks.append(mask)
        tokens = torch.cat(tokens, dim=0)
        masks = torch.cat(masks, dim=0)
        return tokens, masks

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # tokens and mask is not used during training, only for debug
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        img_id = torch.tensor(int(self.image_ids[self.caption2embedding[item]]), dtype=torch.int64).unsqueeze(0)
        prefix = prefix.float()
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)
        return img_id, (tokens, mask, prefix)

    def __init__(self, gt_path: str, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=True, mode: str="train"):
        self.gt_file = gt_path
        self.mode = mode
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Image Embedding Data size for generation is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        print("Clip embedding dim is %0d" % self.prefixes.shape[1])
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append([torch.tensor(self.tokenizer.encode(cap), dtype=torch.int64) for cap in caption['caption']])
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, max([len(cap) for cap in self.captions_tokens[-1]]))
            # self.max_seq_len = max_seq_len
            if is_main_process():
                with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                    pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
            synchronize()
        all_len = torch.tensor([len(cap) for i in range(len(self)) for cap in self.captions_tokens[i]]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class PairsFeatures(Dataset):
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        data = self.data[item]
        image = data["image"]
        caption = data["caption"]
        image, caption = image.float(), caption.float()
        return image, caption

    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Pair data size is %0d" % len(all_data))
        self.data = all_data
        print("Image embedding dim is %0d" % all_data[0]["image"].shape[0])
        print("Caption embedding dim is %0d" % all_data[0]["caption"].shape[0])
        sys.stdout.flush()


class TextFeatures(Dataset):
    def __len__(self) -> int:
        return len(self.corpus_tokens) 

    def pad_tokens(self, item: int):
        tokens = self.corpus_tokens[item]
        if tokens.shape[0] > self.max_length:
            tokens[self.max_length-2] = self.tokenizer.encode('.')[0]
            tokens[self.max_length-1] = tokens[-1]
        padding = self.max_length - tokens.shape[0]
        if padding > 0:
            # for roberta padding is 1
            tokens = torch.cat((tokens, torch.ones(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_length]
        mask = ~tokens.eq(1)  # mask is zero where we out of sequence
        tokens[~mask] = 1
        mask = mask.float()
        return tokens, mask

    def __getitem__(self, item: int) -> torch.Tensor:
        token_list = []
        mask_list = []
        for i in range(item*self.sample_times, (item+1)*self.sample_times):
            i %= len(self.corpus_tokens)
            if item != 0 and i == 0:
                self.shuffle()
            tokens, mask = self.pad_tokens(i)
            token_list.append(tokens)
            mask_list.append(mask)
        return (torch.stack(token_list, dim=0), torch.stack(mask_list, dim=0))

    def __init__(self, sample_times: int, corpus_path: str, max_length: int, discriminator_type: str):
        """
        corpus_length : file path
        max_length: corpus sentences max length
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(discriminator_type)
        ### consider <s> and </s> for roberta
        self.max_length = max_length+2
        self.sample_times = sample_times
        with open(corpus_path, 'rb') as f:
            corpus_data = pickle.load(f)
        print("Text Corpus size for Discrimination is %0d" % len(corpus_data))
        sys.stdout.flush()
        if os.path.isfile(f"{corpus_path[:-4]}_tokens.pkl"):
            with open(f"{corpus_path[:-4]}_tokens.pkl", 'rb') as f:
                self.corpus_tokens = pickle.load(f)
        else:
            self.corpus_tokens = []
            for sentence in corpus_data:
                self.corpus_tokens.append(torch.tensor(self.tokenizer.encode(sentence), dtype=torch.int64))
            # self.max_seq_len = max_seq_len
            if is_main_process():
                with open(f"{corpus_path[:-4]}_tokens.pkl", 'wb') as f:
                    pickle.dump(self.corpus_tokens, f)
            synchronize()
    
    def shuffle(self):
        random.shuffle(self.corpus_tokens)
        return 0

####### Models
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act='tanh'):
        super(MLP, self).__init__()
        if act == 'tanh':
            act = nn.Tanh
        elif act == 'gelu':
            act = nn.GELU
        elif act == 'relu':
            act = nn.ReLU
        else:
            act = nn.Tanh
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    def initialization(self, init_method: str = "normal", sigma: float = 0.1):
        def normal_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight.data, 0, sigma)
                nn.init.zeros_(m.bias.data)
        if init_method == "normal":
            self.model.apply(normal_init_weights)


class Mapping_Network(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act='tanh'):
        super(Mapping_Network, self).__init__()
        self.model = MLP(sizes, bias, act)

    def initialization(self, init_method: str = "normal", sigma: float = 0.1):
        self.model.initialization(init_method, sigma)

    def forward(self, x: torch.Tensor, training=True) -> torch.Tensor:
        if training:
            self.model.train()
            return x + self.model(x)
        else:
            self.model.eval()
            with torch.no_grad():
                return x + self.model(x)
       
class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention
    


class TransformerLayer(nn.Module):
    """
        implement as Pre-LN:ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE
    """
    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):
    
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: str = "mlp"):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == "mlp":
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
            print("Using MLP as Mapper")
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)
            print("Using Transformer as Mapper")


    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        if is_decode:
            return self.generate_cap(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)    

    def encode_forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out            

    def generate_cap(self, img_feats, max_length=None,
            do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, eos_token=None, length_penalty=None,
            num_return_sequences=None, num_keep_best=1, is_decode=None
            ):
        """ Generates captions given image features
        """
        # TODO: self.num_keep_best is given in generate, then search func should not use magic number 
        assert is_decode
        batch_size = img_feats.shape[0]
        self.num_keep_best = num_keep_best

        ## project prefix to GPT space
        # [bs, pre_len, emb_size]
        img_feats = self.clip_project(img_feats).view(-1, self.prefix_length, self.gpt_embedding_size)

        cur_len = 0
        if  num_return_sequences != 1:
            # Expand input to num return sequences
            img_feats = self._expand_for_beams(img_feats, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        output = self.generate(
            img_feats,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            eos_token,
            effective_batch_size,
        )

        return output

    def generate(
            self,
            input_embeds,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            eos_token,
            batch_size,
        ):
            """ Generate sequences for each example without beam search (num_beams == 1).
                All returned sequence are generated independantly.
            """
            self.num_keep_best = 1
            assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
            # current position / max lengths / length of generated sentences / unfinished sentences
            unfinished_sents = []
            #[bs] = 1
            cur_unfinished = input_embeds.new(batch_size).fill_(1)
            eos_token_id = self.tokenizer.encode(eos_token)[0]

            # log of scores for each sentence in the batch
            logprobs = []

            past = None

            token_ids = None

            pad_token_id = self.tokenizer.encode('<|endoftext|>')[0]

            while cur_len < max_length:
                if past:
                    outputs = self.gpt(input_ids=next_token.unsqueeze(-1), past_key_values=past)
                else:
                    outputs = self.gpt(inputs_embeds=input_embeds, past_key_values=past)
                # [bs, pre_len, 768] -> [bs, voc_size]
                next_token_logits = outputs.logits[:, -1, :]

                # if model has past, then set the past variable to speed up decoding
                if self._do_output_past(outputs):
                    past = outputs.past_key_values

                # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                if repetition_penalty != 1.0 and token_ids is not None:
                    for i in range(batch_size):
                        for previous_token in set(token_ids[i].tolist()):
                            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                            if next_token_logits[i, previous_token] < 0:
                                next_token_logits[i, previous_token] *= repetition_penalty
                            else:
                                next_token_logits[i, previous_token] /= repetition_penalty

                if do_sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    # Top-p/top-k filtering
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    # Sample
                    # [bs * sample_time]
                    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    # next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.argmax(next_token_logits, dim=-1)

                # Compute scores
                _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
                _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
                logprobs.append(_scores)  # (batch_size, 1)
                unfinished_sents.append(cur_unfinished)

                tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
                if token_ids is None:
                    token_ids = tokens_to_add.unsqueeze(-1)
                else:
                    token_ids = torch.cat([token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                

                #for t in input_ids:
                    #print(self.tokenizer.convert_ids_to_tokens(t.tolist()))
                cur_unfinished = cur_unfinished.mul(next_token.ne(eos_token_id).long())
                cur_len = cur_len + 1

                # stop when there is a </s> in each sentence, or if we exceed the maximul length
                if cur_unfinished.max() == 0:
                    break

            # add eos_token_ids to unfinished sentences
            # NOTE: for OSCAR pretrained model, it should be ended with SEP token. However we end with '.' to keep consistent with OSCAR
            if cur_len == max_length:
                token_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_id)

            logprobs = torch.cat(logprobs, dim=1)
            unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
            sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
            # return logprobs to keep consistent with beam search output
            logprobs = sum_logprobs / unfinished_sents.sum(dim=1)
            # (batch_size, n_best, max_len), (batch_size, n_best)

            pad_len = max_length - token_ids.shape[1]
            if pad_len > 0:
                padding_ids = token_ids.new(batch_size, pad_len).fill_(pad_token_id)
                token_ids = torch.cat([token_ids, padding_ids], dim=1)

            # (batch_size, n_best, max_len), (batch_size, n_best)
            return token_ids.unsqueeze(1), logprobs.unsqueeze(1)

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x
        # x: [bs, len, embed_size]
        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        # expanded_x: batch * num_expand * len * embed_size
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        # x: (batch * num_expand) * len * embed_size
        return x

    def _do_output_past(self, outputs):
        return len(outputs) > 1


class ScstRewardCLIPCriterion(torch.nn.Module):
    REWARD_WEIGHT = 1
    def __init__(self, args, baseline_type='greedy'):
        # TODO: text corpus dicriminator
        super().__init__()
        self.ClipRewarder = ClipDiscriminator(args)
        assert baseline_type in ['greedy', 'sample']
        self.baseline_type = baseline_type
        self._cur_score = None
        self.batch_size = None
        self.text_features = None

    def forward(self, img_feats, gt_text, greedy_res, sample_res, sample_logprobs):
        batch_size = len(img_feats)
        self.batch_size = batch_size
        self._cur_score = None
        self.text_features = None
        sample_res_size = len(sample_res)
        seq_per_img = sample_res_size // batch_size

        gen_res = []
        gen_res.extend(sample_res)
        gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        if self.baseline_type == 'greedy':
            assert len(greedy_res) == batch_size
            gen_res.extend(greedy_res)
            gt_idx.extend([i for i in range(batch_size)])

        scores = self._clip_eval_scores(gen_res, gt_idx, img_feats, gt_text)

        if self.baseline_type == 'greedy':
            baseline = scores[-batch_size:][:, np.newaxis]
        else:
            sc_ = scores.reshape(batch_size, seq_per_img)
            baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        # sample - baseline
        reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        self._cur_score = reward.mean()
        reward = reward - baseline
        reward = reward.reshape(sample_res_size)

        reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        loss = - sample_logprobs * reward
        loss = loss.mean()
        return loss

    def get_score(self):
        l1_score, cos_score, clip_score = self.ClipRewarder.get_score()
        return l1_score, cos_score, clip_score

    def _clip_eval_scores(self, gen_res, gt_idx, image_embeds, text_embeds):
        gen_res_size = len(gen_res)

        res = []
        for i in range(gen_res_size):
            res.append(self._wrap_sentence(gen_res[i]))

        image_features = []
        for i in range(gen_res_size):
            image_features.append(image_embeds[gt_idx[i]])

        text_features = []
        for i in range(gen_res_size):
            text_features.append(text_embeds[gt_idx[i]])

        scores = self.ClipRewarder(res, image_features, text_features, self.batch_size)
        self.text_features = self.ClipRewarder.get_text_features()
        scores = self.REWARD_WEIGHT * scores
        return scores

    def get_text_features(self):
        return self.text_features

    @classmethod
    def _wrap_sentence(self, s):
        # ensure the sentence ends with <eos> token
        # in order to keep consisitent with cider_cached_tokens
        r = s.strip()
        if r.endswith('.'):
            r = r[:-1]
        # NOTE: for CLIP text encoder, should not add <eos> token
        # r += ' <eos>'
        return r

class RobertaDiscriminator(nn.Module):

    def __init__(self, args, baseline_type, discriminator_type:str):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained(discriminator_type)
        self.tokenizer = RobertaTokenizer.from_pretrained(discriminator_type)
        self.mlp = MLP((768, 384, 1))
        self.criterion = nn.BCEWithLogitsLoss()
        assert baseline_type in ['greedy', 'sample']
        self.baseline_type = baseline_type
        self._cur_score = None
        ## +2 for <s> and </s>
        self.max_length = args.max_gen_length + 2
        self.device = args.device

    def forward(self, tokens, mask, training=False, labels=None):
        if training:
            self.train()
            Res = self.text_encoder(input_ids = tokens, attention_mask = mask)
            pooler_out = Res[1]
            logits = self.mlp(pooler_out)
            labels = labels.to(self.device)
            loss = self.criterion(logits.squeeze(), labels)
            return loss
        else:
            with torch.no_grad():
                self.eval()
                Res = self.text_encoder(input_ids = tokens, attention_mask = mask)
                pooler_out = Res[1]
                logits = self.mlp(pooler_out)
                self.train()
                return logits

    def get_score(self):
        score = self._cur_score.item()
        return score

    def pad_and_token(self, gen_res):
        ress = []
        masks = []
        for s in gen_res:
            tokens = torch.tensor(self.tokenizer.encode(s), dtype=torch.int64)
            if tokens.shape[0] > self.max_length:
                tokens[self.max_length-2] = self.tokenizer.encode('.')[0]
                tokens[self.max_length-1] = tokens[-1]
            padding = self.max_length - len(tokens)
            if padding > 0:
            # for roberta padding is 1
                tokens = torch.cat((tokens, torch.ones(padding, dtype=torch.int64)))
            elif padding < 0:
                tokens = tokens[:self.max_length]
            mask = ~tokens.eq(1)  # mask is zero where we out of sequence
            tokens[~mask] = 1
            mask = mask.float()
            masks.append(mask)
            ress.append(tokens)
        ress = torch.stack(ress, dim=0)
        ress = ress.to(self.device)
        masks = torch.stack(masks, dim=0)
        masks = masks.to(self.device)
        return ress, masks

    def PG_loss(self, greedy_res, sample_res, sample_logprobs):
        batch_size = len(greedy_res)
        sample_res_size = len(sample_res)
        seq_per_img = sample_res_size // batch_size

        gen_res = []
        gen_res.extend(sample_res)
        gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        if self.baseline_type == 'greedy':
            assert len(greedy_res) == batch_size
            gen_res.extend(greedy_res)
            gt_idx.extend([i for i in range(batch_size)])

        ress, masks = self.pad_and_token(gen_res)

        scores = self.forward(ress, masks).squeeze()

        if self.baseline_type == 'greedy':
            baseline = scores[-batch_size:][:, np.newaxis]
        else:
            sc_ = scores.reshape(batch_size, seq_per_img)
            baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        # sample - baseline
        reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        self._cur_score = reward.mean()
        reward = reward - baseline
        reward = reward.reshape(sample_res_size)

        reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        loss = - sample_logprobs * reward
        loss = loss.mean()
        return loss

class ClipDiscriminator(torch.nn.Module):
    """
    TODO: negative sampling
    """
    def __init__(self, args):
        super().__init__()
        self.clip_model, preprocess = clip.load(args.clip_model_type, device=args.device, jit=False)
        self.text_features = None
        self.args = args
        self.l1loss = torch.nn.L1Loss(reduction='none')
        self.cos_score = None
        self.l1_score = None
        self.clip_score = None

    def forward(self, text, image_features, gt_features, bs):
        self.text_features = None
        with torch.no_grad():
            try:
                text = clip.tokenize(text).to(self.args.device)
            except:
                text = clip.tokenize(text, truncate=True).to(self.args.device)
            text_features = self.clip_model.encode_text(text)
            self.text_features = text_features
            # if is_training:
            #     text_features = all_gather(text_features, self.args)
            #     image_features = all_gather(image_features, self.args)
            #     torch.distributed.barrier()
            image_features = torch.vstack(image_features)
            gt_features = torch.vstack(gt_features)
            logit_scale = self.clip_model.logit_scale.exp()

            loss = self.l1loss(gt_features, text_features).mean(axis=1)
            l1_score = logit_scale * (1 - loss)
            self.l1_score = l1_score[:-bs].mean().item()


            gt_features = gt_features / gt_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logits_per_image = logit_scale * gt_features @ text_features.t().type_as(gt_features)
            batch_size = gt_features.shape[0]
            diag_ind = np.arange(batch_size)
            cos_score = logits_per_image[diag_ind, diag_ind]
            self.cos_score = cos_score[:-bs].mean().item()
 
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            clip_logits = logit_scale * image_features @ text_features.t().type_as(image_features)
            clip_score = clip_logits[diag_ind, diag_ind]
            self.clip_score = clip_score[:-bs].mean().item()

            cos_w = self.args.cos_weight
            clip_w = self.args.clip_weight

            sim_score = cos_w * cos_score + (1 - cos_w) * l1_score
            score =  (1 - clip_w) * sim_score + clip_w * clip_score

            return score

    def get_score(self):
        return self.l1_score, self.cos_score, self.clip_score



    def get_text_features(self):
        if self.args.normalize_prefix:
            text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        else:
            text_features = self.text_features
        text_features = text_features.to(dtype=torch.float32)
        text_features = text_features.to(self.args.device)
        return text_features

