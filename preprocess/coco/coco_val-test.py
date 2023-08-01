import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str, coco_path: str):
    device = torch.device('cuda')
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    mode_list = ['val', 'test']
    for mode in mode_list:
        out_path = coco_path + f"/coco_{clip_model_name}_{mode}.pkl"
        json_path = coco_path + f"/annotations/{mode}_caption_coco_format.json"
        print("saving path:{}".format(out_path))
        with open(json_path, 'r') as f:
            data = json.load(f)
        print("%0d captions loaded from json " % len(data['images']))
        
        all_embeddings = []
        embedding_id2image_id = []

        for i in tqdm(range(len(data['images']))):
            d = data["images"][i]
            img_id = d["id"]
            if mode == 'test':
                another_mode = 'val'
            else:
                another_mode = 'test'
            filename = coco_path + f"/{mode}2014/COCO_{mode}2014_{int(img_id):012d}.jpg"
            if not os.path.isfile(filename):
                filename = coco_path + f"/{another_mode}2014/COCO_{another_mode}2014_{int(img_id):012d}.jpg"
                if not os.path.isfile(filename):
                    raise IndexError
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            all_embeddings.append(prefix)
            embedding_id2image_id.append(img_id)

        all_captions = [{"image_id":embedding_id2image_id[i], "clip_embedding":i, "caption":[]} for i in range(len(embedding_id2image_id))]

        for i in tqdm(range(len(data['annotations']))):
            d = data['annotations'][i]
            img_id = d["image_id"]
            all_captions[embedding_id2image_id.index(img_id)]["caption"].append((data['annotations'][i]['caption']))

        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

        print('Done')
        print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('ViT-L/14', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--coco_path', default="./data/coco", type=str, help="path to coco root directory")
    args = parser.parse_args()
    main(args.clip_model_type, args.coco_path)