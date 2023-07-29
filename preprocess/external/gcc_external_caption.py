import pickle
import pickle as pkl
from tqdm import tqdm
import json
import string
import numpy as np
import torch
import skimage.io as io
import clip
from PIL import Image
import os
import argparse
"""
do 3 things:
1. Judge the first token and the last token is punctuation or not ? if TRUE remove first token, replace last punctuation with '.'
2. Cut from first '.'
3. Remove length <= 3 sentence
"""


def main(clip_model_type: str, tsv_path: str, out_path: str):
    additional_punc = {"''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
            ".", "?", "!", ",", ":", "-", "--", "...", ";"}
    python_default_func = set(string.punctuation)
    punc_set = additional_punc | python_default_func
    f = open(tsv_path, encoding="utf-8", errors="ignore")
    discriptions = [line.split("\t")[0] for line in f]
    device = torch.device('cuda')
    sentences = []
    sentences_len = [0 for i in range(100)]
    for discription_ in tqdm(discriptions):
        ### remove first punctuation and last punctuation
        discription = discription_.split()
        if discription[0] in punc_set:
            discription = discription[1:]
        if discription[-1] in punc_set:
            discription = discription[:-1]
        discription.append('.')
        ### locate '.'
        end = discription.index('.')
        discription = discription[:end]
        length = len(discription)

        np.random.seed(0)
        if length <= 6:
            #print(discription)
            continue
        elif length > 15 and length < 20:
            rand = np.random.rand()
            if rand >= 0.3:
                continue
        elif length >= 20:
            continue
    
        if length < 100:
            sentences_len[length-1] += 1
        else:
            sentences_len[-1] += 1
        caption = ' '.join(discription)
        # caption add '.' and Case first word
        try:
            caption = str.upper(caption[0]) + caption[1:]
        except:
            print('cap invalid')
            continue
        caption += '.'
        sentences.append(caption)

    print(sentences_len)

    np.random.seed(1024)
    np.random.shuffle(sentences)

    sentence_path = out_path + "/gcc_external_sentences.pkl"
    print("shuffled sentences saving path:{}".format(sentence_path))
    with open(sentence_path, 'wb') as f:
        pickle.dump(sentences, f)    

    data = sentences
    clip_model_name = clip_model_type.replace('/', '_')
    out_path += f"/gcc_{clip_model_name}_external_captions.pkl"
    print("saving path:{}".format(out_path))
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    data_size = len(data)
    index = [i for i in range(data_size)]
    np.random.seed(1024)
    np.random.shuffle(index)
    print("%0d captions loaded from pkl" % len(data))
    all_embeddings = []
    embedding_id2image_id = []

    for i in tqdm(index):
        img_id = str(i)
        with torch.no_grad():
            assert data[i][-1] == '.'
            text = clip.tokenize(data[i][:-1], truncate=True).to(device)
            prefix = clip_model.encode_text(text).cpu()
        all_embeddings.append(prefix)
        embedding_id2image_id.append(img_id)

    all_captions = [{"image_id":embedding_id2image_id[i], "clip_embedding":i, "caption":''} for i in range(len(embedding_id2image_id))]

    for i in tqdm(index):
        all_captions[embedding_id2image_id.index(str(i))]["caption"] = data[i]

    print("Saving...")
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-L/14", choices=('ViT-L/14', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--data_path', default="./data/external/gcc/Train_GCC-training.tsv", type=str, help="gcc tsv file path")
    parser.add_argument('--out_path', default="./data/external/gcc", type=str, help="output folder path")
    args = parser.parse_args()
    main(args.clip_model_type, args.data_path, args.out_path)