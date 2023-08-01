import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import math
"""
to parse train_
Unsupervised data preprocess for training, index by img not caption
"""

def main(clip_model_type: str, out_path: str, img_path: str, cap_path: str, img_data: str, cap_data: str, t: int):
    clip_model_name = clip_model_type.replace('/', '_')
    out_path += "/{}_images_{}_captions_{}_{}.pkl".format(img_data, cap_data, clip_model_name, t)
    print("saving path:{}".format(out_path))

    with open(img_path, 'rb') as f:
        img_data = pickle.load(f)
    with open(cap_path, 'rb') as f:
        cap_data = pickle.load(f)

    device = torch.device('cuda')

    if clip_model_type == "ViT-L/14":
        dim = 768
    elif clip_model_type == "RN50x4":
        dim = 640
    elif clip_model_type == "ViT-B/32":
        dim = 512

    img_data = img_data["clip_embedding"].to(device)
    cap_data = cap_data["clip_embedding"].to(device)
    print("%0d images loaded from json " % len(img_data))
    print("%0d captions loaded from json " % len(cap_data))

    img_data_norm = img_data / img_data.norm(dim=1, keepdim=True)
    cap_data_norm = cap_data / cap_data.norm(dim=1, keepdim=True)
    cap_data_norm = cap_data_norm.to(torch.float64)
    assert img_data.shape[1] == dim
    assert cap_data.shape[1] == dim
    
    paired_embedding = []

    for i in tqdm(range(len(img_data))):
        with torch.no_grad():
            img_emb = img_data[i]
            img_emb_norm = img_data_norm[i].reshape(1, -1).to(torch.float64)
            similarity_matrix = (img_emb_norm * cap_data_norm).sum(-1)
            weight_matrix = torch.exp(similarity_matrix * t)
            weight_sum = weight_matrix.sum().item()
            if weight_sum == float('nan'):
                raise RuntimeError('Overflow error')        
            weight_matrix = (weight_matrix / weight_sum).reshape(-1, 1)
            generate_emb = (cap_data * weight_matrix).sum(0)
            generate_emb = generate_emb.to(torch.float16).to('cpu')
            img_emb = img_emb.to(torch.float16).to('cpu')
            dict = {"image": img_emb, "caption": generate_emb}
            paired_embedding.append(dict)
            
    print(len(paired_embedding))
    print("Saving...")
    with open(out_path, 'wb') as f:
        pickle.dump(paired_embedding, f)
    print('Done')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-L/14", choices=('ViT-L/14', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--output_dir', default="./data/coco", type=str, help="output folder path")
    parser.add_argument('--image_pkl', required=True, type=str, help="path to image embedding pkl")
    parser.add_argument('--caption_pkl', required=True, type=str, help="path to caption embedding pkl")
    parser.add_argument('--image_dataset', default="coco", type=str, help="image dataset name")
    parser.add_argument('--caption_corpus', default="coco", type=str, help="caption corpus name")
    parser.add_argument('--t', default=100, type=int, help="Temperature")
    args = parser.parse_args()
    main(args.clip_model_type, args.output_dir, args.image_pkl, args.caption_pkl, args.image_dataset, args.caption_corpus, args.t)