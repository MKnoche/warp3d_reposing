# Reposing Humans by Warping 3D Features

**[Project](https://www.vision.rwth-aachen.de/publication/00201/) | [Paper](https://arxiv.org/pdf/2006.04898.pdf) |
[Video](https://www.youtube.com/watch?v=U4hfTcF2cHI)**

![alt text](teaser.gif) 

This repository contains code for our paper “Reposing Humans by Warping 3D Features”. We warp implicitely learned
volumetric features with explicit 3D transformations to change the pose of a given person into any desired target pose.

## Setup
Using conda:
```
conda create -n warp3d_reposing
conda activate warp3d_reposing
conda install tensorflow-gpu=1.12
conda install scikit-image
conda install -c menpo opencv3
conda install numpy=1.16
```

## Code Overview

* `parameters.py` contains global parameters for training, model and dataset
* `dataset.py` loads pairs of images and their respective poses, estimates the transformations and creates the
bodypart-masks
* `model.py` contains generator and discriminator as well as the warping block
* `train.py` glues everything together

## Datasets

The parameter `params['data_dir']` points to the root directory which includes all dataset directories. Each dataset
directory contains a directory `images` with all images organized in subfolders. It further must include a pickle file
`poses.pkl` containing nested dictionaries with the same hierarchy as the image directory. Each image `img.png` is
replaced by a dictionary key `img` which points to a numpy array with one row per joint and three columns for the
coordinates in pixel space. Height and width are between 0 and 256, the depth is centered around 0.

### Estimated Poses

[poses.pkl](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/fashion3d/poses.pkl) for deepfashion, save as
`<data_dir>/fashion3d/poses.pkl`

[poses.pkl](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/iPER/poses.pkl) for iPER, save as
`<data_dir>/iPER/poses.pkl`

## Training

First, adapt the paths for the data root directory, the checkpoint directory and the tensorboard directory in
`parameters.py` according to your system structure.

Second, download the
[pose estimator checkpoint](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/pose3d_minimal/checkpoint.zip)
which is used for validation. Extract the three files into `pose3d_minimal/checkpoint/`.

You now have two options to start a training. You can run `train.py` with command-line parameters, for example

```python3 train.py name feat_weight_5 feature_loss_weight 5```.

Make sure to always include a unique name, because a new directory with this name will be created in the checkpoint
directory and the tensorboard directory.

To keep track of different parameter combinations, you can also define a
`JOB_ID` in `parameters.py` and then for example use

```python3 train.py JOB_ID 5```.

For each ablation model of our paper, a `JOB_ID` is already defined.

## Pretrained Models

Extract the checkpoints into `<check_dir>/<model_name>/`.

### iPER

| name               | 3D warping         | 3D target pose     | checkpoint                                                                                                       |
| ------------------ | ------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **iPER-3d_w-3d_p** | :heavy_check_mark: | :heavy_check_mark: | [iPER-3d_w-3d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/iPER-3d_w-3d_p.zip) |
|   iPER-3d_w-2d_p   | :heavy_check_mark: | :heavy_minus_sign: | [iPER-3d_w-2d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/iPER-3d_w-2d_p.zip) |
|   iPER-2d_w-3d_p   | :heavy_minus_sign: | :heavy_check_mark: | [iPER-2d_w-3d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/iPER-2d_w-3d_p.zip) |
|   iPER-2d_w-2d_p   | :heavy_minus_sign: | :heavy_minus_sign: | [iPER-2d_w-2d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/iPER-2d_w-2d_p.zip) |

### deepfashion

| name               | 3D warping         | 3D target pose     | checkpoint                                                                                                       |
| ------------------ | ------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **fash-3d_w-3d_p** | :heavy_check_mark: | :heavy_check_mark: | [fash-3d_w-3d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/fash-3d_w-3d_p.zip) |
|   fash-3d_w-2d_p   | :heavy_check_mark: | :heavy_minus_sign: | [fash-3d_w-2d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/fash-3d_w-2d_p.zip) |
|   fash-2d_w-3d_p   | :heavy_minus_sign: | :heavy_check_mark: | [fash-2d_w-3d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/fash-2d_w-3d_p.zip) |
|   fash-2d_w-2d_p   | :heavy_minus_sign: | :heavy_minus_sign: | [fash-2d_w-2d_p.zip](https://omnomnom.vision.rwth-aachen.de/data/warp3d_reposing/checkpoints/fash-2d_w-2d_p.zip) |

## BibTeX

```
@inproceedings{Knoche20CVPRW,
 author = {Markus Knoche and Istv\'an S\'ar\'andi and Bastian Leibe},
 title = {Reposing Humans by Warping 3{D} Features},
 booktitle = {CVPR Workshop on Towards Human-Centric Image/Video Synthesis},
 year = {2020}
}
```
