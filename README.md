# GeoDiffusion
[![arXiv](https://img.shields.io/badge/arXiv-2306.04607-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2306.04607) [![arXiv](https://img.shields.io/badge/Web-GeoDiffusion-blue.svg?style=plastic)](https://kaichen1998.github.io/projects/geodiffusion/)

This repository contains the implementation of the paper:

> GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation <br>
> [Kai Chen](https://kaichen1998.github.io/), [Enze Xie](https://xieenze.github.io/), [Zhe Chen](https://czczup.github.io/), [Yibo Wang](https://openreview.net/profile?id=~Yibo_Wang7), [Lanqing Hong](https://scholar.google.com/citations?hl=zh-CN&user=2p7x6OUAAAAJ&view_op=list_works&sortby=pubdate), [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ&hl=zh-CN), [Dit-Yan Yeung](https://sites.google.com/view/dyyeung/home) <br>
> *International Conference on Learning Representations (ICLR), 2024.*

![img](./images/overview.png)



## Installation

Clone this repo and create the GeoDiffusion environment with conda. We test the code under `python==3.7.16, pytorch==1.12.1, cuda=10.2` on Tesla V100 GPU servers. Other versions might be available as well.

1. Initialize the conda environment:

   ```bash
   git clone https://github.com/KaiChen1998/GeoDiffusion.git
   conda create -n geodiffusion python=3.7 -y
   conda activate geodiffusion
   ```

2. Install the required packages:

   ```bash
   cd GeoDiffusion
   # when running training
   pip install -r requirements/train.txt
   # only when running inference with DPM-Solver++
   pip install -r requirements/dev.txt
   ```



## Download Pre-trained Models

|        Dataset        | Image Resolution | Grid Size |                           Download                           |
| :-------------------: | :--------------: | :-------: | :----------------------------------------------------------: |
|       nuImages        |     256x256      |  256x256  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-nuimages-256x256) |
|       nuImages        |     512x512      |  512x512  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-nuimages-512x512) |
| nuImages_time_weather |     512x512      |  512x512  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-nuimages-time-weather-512x512) |
|      COCO-Stuff       |     256x256      |  256x256  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-coco-stuff-256x256) |
|      COCO-Stuff       |     512x512      |  256x256  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-coco-stuff-512x512) |




## Detection Data Generation with GeoDiffusion

Download the pre-trained models and put them under the root directory. Run the following commands to run detection data generation with GeoDiffusion. For simplicity, we embed the layout definition process in the file `run_layout_to_image.py` directly. Check [here](./run_layout_to_image.py#L75-L82) for detailed definition.

```bash
python run_layout_to_image.py $CKPT_PATH --output_dir ./results/
```



## Train GeoDiffusion

### 1. Prepare dataset

We primarily use the [nuImages](https://www.nuscenes.org/nuimages) and [COCO-Stuff](https://cocodataset.org/#home) datasets for training GeoDiffusion. Download the image files from the official websites. For better training performance, we follow [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/nuimages/README.md/#introduction) to convert the nuImages dataset into COCO format (you can also download our converted annotations via [HuggingFace](https://huggingface.co/datasets/KaiChen1998/nuimages-geodiffusion)), while the converted annotation file for COCO-Stuff can be download via [HuggingFace](https://huggingface.co/datasets/KaiChen1998/coco-stuff-geodiffusion). The data structure should be as follows after all files are downloaded.

```
├── data
│   ├── coco
│   │   │── coco_stuff_annotations
│   │   │   │── train
│   │   │   │   │── instances_stuff_train2017.json
│   │   │   │── val
│   │   │   │   │── instances_stuff_val2017.json
│   │   │── train2017
│   │   │── val2017
│   ├── nuimages
│   │   │── annotation
│   │   │   │── train
│   │   │   │   │── nuimages_v1.0-train.json
│   │   │   │── val
│   │   │   │   │── nuimages_v1.0-val.json
│   │   │── samples
```

### 2. Launch distributed training

We use [Accelerate](https://huggingface.co/docs/accelerate/index) to launch efficient distributed training (with 8 x V100 GPUs by default). We encourage readers to check the official documents for personalized training settings. We provide the default training parameters in this [script](./tools/dist_train.sh), and to change the training dataset, we can directly change the `dataset_config_name` argument.

```bash
# COCO-Stuff
bash tools/dist_train.sh \
	--dataset_config_name configs/data/coco_stuff_256x256.py \
	--output_dir work_dirs/geodiffusion_coco_stuff

# nuImages
bash tools/dist_train.sh \
	--dataset_config_name configs/data/nuimage_256x256.py \
	--output_dir work_dirs/geodiffusion_nuimages
```

We also support continuing fine-tuning a pre-trained GeoDiffusion checkpoint on downstream tasks to support more geometric controls in the [Textural Inversion](https://arxiv.org/abs/2208.01618) manner by only training the newly added tokens. We encourage readers to check [here](https://github.com/KaiChen1998/GeoDiffusion/blob/main/train_geodiffusion.py#L455) and [here](https://github.com/KaiChen1998/GeoDiffusion/blob/main/train_geodiffusion.py#L488) for more details.

```bash
bash tools/dist_train.sh \
	--dataset_config_name configs/data/coco_stuff_256x256.py \
	--train_text_encoder_params added_embedding \
	--output_dir work_dirs/geodiffusion_coco_stuff_continue
```



### 3. Launch batch inference

Different from the more user-friendly inference demo provided [here](https://github.com/KaiChen1998/GeoDiffusion?tab=readme-ov-file#detection-data-generation-with-geodiffusion), in this section we provide the scripts to run batch inference throughout a dataset. Note that the inference settings might differ for different checkpoints. We encourage readers to check the `generation_config.json` file under each pre-trained checkpoint in the [Model Zoo](https://github.com/KaiChen1998/GeoDiffusion/tree/main?tab=readme-ov-file#download-pre-trained-models) for more details.

```bash
# COCO-Stuff
# We encourage readers to check https://github.com/ZejianLi/LAMA?tab=readme-ov-file#testing
# to report quantitative results on COCO-Stuff L2I benchmark.
bash tools/dist_test.sh PATH_TO_CKPT \
	--dataset_config_name configs/data/coco_stuff_256x256.py

# nuImages
bash tools/dist_test.sh PATH_TO_CKPT \
	--dataset_config_name configs/data/nuimage_256x256.py
```



## Qualitative Results

More results can be found in the main paper.

![img](./images/qualitative_1.PNG)

![img](./images/qualitative_2.PNG)



## The GeoDiffusion Family

We aim to construct a controllable and flexible pipeline for perception data corner case generation and visual world modeling! Check our latest works:

- [GeoDiffusion](https://kaichen1998.github.io/projects/geodiffusion/): text-prompted geometric controls for 2D object detection.
- [MagicDrive](https://gaoruiyuan.com/magicdrive/): multi-view street scene generation for 3D object detection.
- [TrackDiffusion](https://kaichen1998.github.io/projects/trackdiffusion/): multi-object video generation for MOT tracking.
- [DetDiffusion](https://arxiv.org/abs/2403.13304): customized corner case generation.
- [Geom-Erasing](https://arxiv.org/abs/2310.05873): geometric controls for implicit concept removal.



## Citation

```bibtex
@article{chen2023integrating,
  author    = {Chen, Kai and Xie, Enze and Chen, Zhe and Hong, Lanqing and Li, Zhenguo and Yeung, Dit-Yan},
  title     = {Integrating Geometric Control into Text-to-Image Diffusion Models for High-Quality Detection Data Generation via Text Prompt},
  journal   = {arXiv: 2306.04607},
  year      = {2023},
}
```



## Acknowledgement

We adopt the following open-sourced projects:

- [diffusers](https://github.com/huggingface/diffusers/): basic codebase to train Stable Diffusion models.
- [mmdetection](https://github.com/open-mmlab/mmdetection): dataloader to handle images with various geometric conditions.
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) & [LAMA](https://github.com/ZejianLi/LAMA): data pre-processing of the training datasets.
