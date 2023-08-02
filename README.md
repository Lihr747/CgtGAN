# CLIP-guided text GAN
Code for our paper: [*CgT-GAN: CLIP-guided Text GAN for Image Captioning](https://arxiv.org/abs/2211.09778). (挂出来改一下链接)

All pre-processed data and pretrained models are released in [BaiduPan](https://pan.baidu.com/s/1Og1PPOOdDFw7jMnG0W07Jw?pwd=s5wk).

## Dataset
All data are placed in ~/data as an example.
### MSCOCO Dataset
1. Download COCO images: [train](http://images.cocodataset.org/zips/train2014.zip) & [val/test](http://images.cocodataset.org/zips/val2014.zip) , put train2014 and val2014 folders in
~/data/coco (coco root directory).
2. Download COCO annotations: [annotations](https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_caption.zip), put all json files in ~/data/coco/annotations.

Example of the COCO root directory folder:
```
./data/coco
--train2014
--val2014
--annotations
```
### Flickr30K Dataset
1. Download images: [train/val/test](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), put flickr30k_images folder in ~/data/Flickr30k.
2. Download annotations: [annotations](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip), put all json files in ~/data/Flickr30k/annotations.
### Text Corpus
* ShutterStock : download from [UIC](https://github.com/fengyang0317/unsupervised_captioning/issues/42)
* Google Conceptual Captions : download from [Train_GCC-training.tsv](https://ai.google.com/research/ConceptualCaptions/download)   

Place all externel data in ~/data/externel for subsequent processing.
## Data_Preprocess

We take MSCOCO Dataset and GCC external corpus as an example.
* Extract clip embeddings for COCO images and captions. All extracted embedding pkl files will be saved in ~/data/coco.
    ```
    python preprocess/coco/coco_train_images.py
    python preprocess/coco/coco_train_captions.py
    python preprocess/coco/coco_val-test.py
    ```
* Extract clip embeddings for GCC captions. The pkl files will be saved in ~/data/external.
    ```
    python preprocess/external/gcc_external_captions.py
    ```
* Then generate aggregated textual embeddings.
    ```
    python preprocess/generate_embeddings.py --image_pkl ./data/coco/coco_ViT-L_14_train_images.pkl --caption_pkl ./data/coco/coco_ViT-L_14_train_captions.pkl --image_dataset coco --caption_corpus coco --t 100
   python preprocess/generate_embeddings.py --image_pkl ./data/coco/coco_ViT-L_14_train_images.pkl --caption_pkl ./data/external/gcc_ViT-L_14_external_captions.pkl --image_dataset coco --caption_corpus gcc --t 175
    ```
## Initialization
* Initialize model using COCO Captions:

```
python initialization.py --output_dir path/to/save/folder --data ./data/coco/coco_ViT-L_14_train_captions.pkl
```
* Initialize model using GCC Captions:
```
python initialization.py --output_dir path/to/save/folder --data ./data/external/gcc_ViT-L_14_external_captions.pkl
```
## Training
* Training model under MSCOCO images <-> MSCOCO captions setting:
```
gpus=0,1
CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
--master_port 17527 \
--nproc_per_node 2 cgtgan.py \
--output_dir path/to/save/folder \
--generator_init path/to/init/model.pt \
--data_train ./data/coco/coco_images_coco_captions_ViT-L_14_100.pkl \
--data_val ./data/coco/coco_ViT-L_14_val.pkl \
--data_test ./data/coco/coco_ViT-L_14_test.pkl \
--text_corpus ./data/coco/coco_train_sentences.pkl \
--gt_val ./data/coco/annotations/val_caption_coco_format.json \
--gt_test ./data/coco/annotations/test_caption_coco_format.json \
--do_train \
--epochs 50 \
> coco.out &
```
* Training model under MSCOCO images <-> GCC captions setting:
```
gpus=0,1
mkdir ./output/gcc
CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
--master_port 17528 \
--nproc_per_node 2 cgtgan.py \
--output_dir path/to/save/folder \
--generator_init path/to/init/model.pt \
--data_train ./data/external/coco_images_gcc_captions_ViT-L_14_175.pkl \
--data_val ./data/coco/coco_ViT-L_14_val.pkl \
--data_test ./data/coco/coco_ViT-L_14_test.pkl \
--text_corpus ./data/external/gcc_external_sentences.pkl \
--gt_val ./data/coco/annotations/val_caption_coco_format.json \
--gt_test ./data/coco/annotations/test_caption_coco_format.json \
--do_train \
--epochs 80 \
> gcc.out &
```
## Evaluation
* Test checkpoint on MSCOCO test set:
```
python -u cgtgan.py \
--output_dir path/to/save/folder \
--generator_init path/to/checkpoint/model.pt \
--data_test ./data/coco/coco_ViT-L_14_test.pkl \
--gt_test ./data/coco/annotations/test_caption_coco_format.json \
--do_eval \
```


