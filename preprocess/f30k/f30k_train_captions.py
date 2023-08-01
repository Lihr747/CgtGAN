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


def main(clip_model_type: str, f30k_path: str):
    additional_punc = {"''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
        ".", "?", "!", ",", ":", "-", "--", "...", ";"}
    python_default_func = set(string.punctuation)
    punc_set = additional_punc | python_default_func
    clip_model_name = clip_model_type.replace('/', '_')
    device = torch.device('cuda')
    json_path = f30k_path + f"/annotations/train_caption.json"
    sentence_path = f30k_path + f"/f30k_train_sentences.pkl"
    out_path = f30k_path + f"/f30k_{clip_model_name}_train_captions.pkl"
    f = open(json_path, encoding="utf-8")
    data = json.load(f)
    discriptions = [d['caption'] for d in data]
    sentences = []
    for discription_ in tqdm(discriptions):
        ### remove first punctuation and last punctuation
        if discription_[-1] in punc_set:
            discription_ = discription_[:-1]
        length = len(discription_.split())
        discription_ += '.'
        discription_ = str.upper(discription_[0]) + discription_[1:]
        sentences.append(discription_)

    np.random.seed(1024)
    np.random.shuffle(sentences)   

    print("shuffled sentences saving path:{}".format(sentence_path))
    with open(sentence_path, 'wb') as f:
        pickle.dump(sentences, f) 

    data = sentences
    clip_model_name = clip_model_type.replace('/', '_')
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
    parser.add_argument('--f30k_path', default="./data/Flickr30k", type=str)
    args = parser.parse_args()
    main(args.clip_model_type, args.f30k_path)