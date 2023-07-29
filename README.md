# SCSTClipCap
## DataSet
### Image Set
* COCO image:
    * annotations: download from [OSCAR](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md), i.e., [coco_caption](https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_caption.zip) note: use azcopy for faster speed. 
    * images: [train](http://images.cocodataset.org/zips/train2014.zip) / [val/test](http://images.cocodataset.org/zips/val2014.zip) 
* Flickr30K image:
    * annotations: download from [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/deepimagesent/), i.e., [JSON_format](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip), then convert to same format as coco by `data_preprocess/convert_flickr30k_to_coco_format.py`.
    * images: [train/val/test](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
### Sentence Corpus
* ShutterStock
    * download from [UIC](https://github.com/fengyang0317/unsupervised_captioning/issues/42)
* Google Conceptual Captions
    * download from [GCC - Train_GCC-training.tsv](https://ai.google.com/research/ConceptualCaptions/download)
* MSCOCO & Flickr30k
    * training split of annoations

## Data Pre-processing
### Image Set
* extract training image offline feature
    ```
    python data_preprocess/parse_coco_unsupervised.py
    python data_preprocess/parse_flickr30k_unsupervised.py
    ```
* build testing feature
    ```
    python data_preprocess/parse_coco_val.py
    python data_preprocess/parse_flickr30k_val.py
    ```
* NOTE: remember to change `mode`, `out_path`.
### Sentence Corpus
* build pkl
    * unsupervised
        ```
        python data_preprocess/parse_shutter_external_data.py
        python data_preprocess/parse_GCC_external_data_filter_short_v2.py
        python data_preprocess/parse_MSCOCO_external_data_first_cased.py
        ```
    * unpaired
        ```
        python data_preprocess/parse_MSCOCO_external_data_first_cased_for_unpaired_training.py
        python data_preprocess/parse_flickr30k_external_data_first_cased_for_unpaired_training.py
        ```
## Generator Initialization (Optional)
* select training set
    ```
    data_preprocess/initialization_pretrain_split.py
    ```
    * NOTE: select correct `out_file_name` and `pretrain_corpus`
* generator initialization (MSCOCO <-> SS as an example, val and test on MSCOCO dataset)
    ```bash
    gpus=3

    method_id=init_training_SS_wo_normalize_0.3_1.0

    output_dir=./data1/output/UIC/SS_MSCOCO_init/$method_id

    mkdir $output_dir

    CUDA_VISIBLE_DEVICES=$gpus

    CUDA_VISIBLE_DEVICES=$gpus nohup python -u train_for_init_fix_concate_bug.py \
    --data ./data1/coco/external/shutter_pretrain_split_ViT-B_32_train.pkl \
    --data_val ./data1/coco/oscar_split_ViT-B_32_val.pkl \
    --data_test ./data/coco/oscar_split_ViT-B_32_test.pkl \
    --image_feature_file ./data1/coco/oscar_split_ViT-B_32_train_unsupervised.pkl \
    --output_dir $output_dir --do_train \
    --learning_rate 0.00002 \
    #--warmup_steps 500 \
    --evaluate_during_training \
    --num_train_epochs 2 \
    --save_steps 400 \
    --lambda1 0.3 \
    --lambda2 1.0 \
    --sigma 0.1 \
    > $output_dir/nohup.out &
    ```

## GAN Training
* MSCOCO <->
```bash
gpus=3,5,6,7

method_id=SS_MSCOCO_constant_sch_1000_linear_clip_0.5_with_MMD_init_ckpt_2400_no_cat_bug # not using beam search

output_dir=./data1/output/UIC/SS_MSCOCO/$method_id

mkdir $output_dir

CUDA_VISIBLE_DEVICES=$gpus

CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.launch \
--master_port 29502 \
--nproc_per_node 4 SCST_caption_with_init_fix_concate_bug.py \
--data ./data1/coco/oscar_split_ViT-B_32_train_unsupervised.pkl \
--data_val ./data1/coco/oscar_split_ViT-B_32_val.pkl \
--data_test ./data1/coco/oscar_split_ViT-B_32_test.pkl \
--external_corpus data/coco/external/shutter_cleaned_sentences.pkl \
--output_dir $output_dir --do_train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 2 \
--scheduler constant \
--learning_rate 0.00001 \
--evaluate_during_training \
--num_train_epochs 6 \
--save_steps 200 \
--use_adv_training \
--gan_warm_steps 1000 \
--generator_init_ckpt data1/output/UIC/SS_MSCOCO_init/init_training_SS_wo_normalize_0.5_0.1/checkpoint-0-2400/model.pt \
--seed 0 \
> $output_dir/nohup.out &
```

## Tools
* plot
    * curve
    * hist
* visualize prefix
    * predict with three distance
* visualize fititious embedding
    * save pkl
    * Clipig
## Acknowledge & Reference
* Clipcap
* OSCAR
* Clipig
