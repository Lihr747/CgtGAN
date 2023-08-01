import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
"""
to parse train_
Unsupervised data preprocess for training, index by img not caption
"""
def main(clip_model_type: str, f30k_path: str):
    device = torch.device('cuda')
    clip_model_name = clip_model_type.replace('/', '_')

    json_path = f30k_path + f"/annotations/train_caption.json"
    out_path = f30k_path + f"/f30k_{clip_model_name}_train_images.pkl"
    print("saving path:{}".format(out_path))
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    
    all_embeddings = []
    embedding_id2image_id = []

    img_ids = set([])
    for d in data:
        img_ids.add(d["image_id"])
    img_ids = list(img_ids)

    for i in tqdm(range(len(img_ids))):
        img_id = img_ids[i]
        filename = f30k_path + f"/flickr30k_images/{int(img_id)}.jpg"
        if not os.path.isfile(filename):
            raise TypeError
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)

    all_captions = [{"image_id":embedding_id2image_id[i], "clip_embedding":i, "caption":[]} for i in range(len(embedding_id2image_id))]

    for i in tqdm(range(len(data))):
        img_id = data[i]["image_id"]
        all_captions[embedding_id2image_id.index(img_id)]["caption"].append((data[i]['caption']))

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
    exit(main(args.clip_model_type, args.f30k_path))