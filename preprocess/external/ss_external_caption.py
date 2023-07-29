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
import string
from re import split as resep

"""
do 3 things:
1. Judge the first token and the last token is punctuation or not ? if TRUE remove first token, replace last punctuation with '.'
2. Cut from first '.'
3. Remove length <= 3 sentence
"""


def main(clip_model_type: str, pkl_path: str, out_path: str):
    device = torch.device('cuda')
    f = open(pkl_path, 'rb')
    discriptions = pkl.load(f)
    np.random.seed(0)
    sentences = []
    sentences_len = [0 for i in range(100)]
    additional_punc = {"''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
            ".", "?", "!", ",", ":", "-", "--", "...", ";"}
    python_default_func = set(string.punctuation)
    punc_set = additional_punc | python_default_func

    sep_pattern = ' : | ; | \. | - '

    def statistics(captions):
        length = []
        length_filtered_punc = []
        for caption in captions:
            tokens = caption.split(' ')
            cap_len = len(tokens)
            length.append(cap_len)
            cnt = 0
            for token in tokens:
                if token not in punc_set:
                    cnt += 1
            length_filtered_punc.append(cnt)
        return length, length_filtered_punc
            
    for discription_ in tqdm(discriptions):
        ### remove <S> and </S>
        discription = discription_[1:-1]
        ### clean first and last punc
        if discription[0] in punc_set:
            discription = discription[1:]
        if discription[-1] in punc_set:
            discription = discription[:-1]
        caption = ' '.join(discription)
        sub_captions = resep(sep_pattern, caption)
        lengths, lengths_filtered_func = statistics(sub_captions)
        for index, (sub_caption, length, length_filtered_func) in enumerate(zip(sub_captions, lengths, lengths_filtered_func)):
            # caption add '.' and Case first word
            if (length_filtered_func <= 5 or length <= 6):
                continue
            else:
                if length < 100:
                    sentences_len[length-1] += 1
                else:
                    sentences_len[-1] += 1
            try:
                sub_caption = str.upper(sub_caption[0]) + sub_caption[1:]
            except:
                print('cap invalid')
                continue
            if index == 0:
                sub_caption += '.'
                rand = np.random.rand()
                if length > 15 and length < 20 and rand >= 0.3:
                    continue
                elif length >= 20  and  rand >= 0.1:
                    continue
                else:
                    sentences.append(sub_caption)
            else:
                break
    print(sentences_len)

    sentence_path = out_path + "/ss_cleaned_external_sentences_v5.pkl"
    print("shuffled sentences saving path:{}".format(sentence_path))
    with open(sentence_path, 'wb') as f:
        pickle.dump(sentences, f)  

    data = sentences
    clip_model_name = clip_model_type.replace('/', '_')
    out_path += f"/ss_{clip_model_name}_external_captions_v5.pkl"
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
    parser.add_argument('--data_path', default="./data/external/ss/shutter_sentences.pkl", type=str, help="ss pkl file path")
    parser.add_argument('--out_path', default="./data/external/ss", type=str, help="output folder path")
    args = parser.parse_args()
    main(args.clip_model_type, args.data_path, args.out_path)