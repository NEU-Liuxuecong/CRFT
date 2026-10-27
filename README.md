# CRFT (CVPR 2026)

### [Project Page](https://github.com/NEU-Liuxuecong/CRFT)

Official implementation of **CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration**.

## Links
- **Code**: [GitHub Repository](https://github.com/NEU-Liuxuecong/CRFT)
- **Paper**: [arXiv](https://arxiv.org/abs/2604.05689)

## Authors

**Xuecong Liu, Mengzhu Ding, Zixuan Sun, Zhang Li, Xichao Teng**

## Overview

<img src="assets/Overview.png">

## Code Update (October 2026)

We have updated the CRFT implementation to improve its usability and facilitate further research.

## High-Resolution Inference Support (October 2026)

We are pleased to release an updated version of CRFT that enables **high-accuracy inference at the original 512 × 512 image resolution**.

In our original CVPR 2026 implementation, CRFT was trained and evaluated at a downsampled resolution of 64 × 64 due to GPU memory constraints.

**With this update, CRFT retains its efficient 64 × 64 training strategy while supporting full-resolution 512 × 512 inference through a newly introduced plug-and-play module.** This enhancement requires no additional training or model retraining, allowing CRFT to achieve excellent registration accuracy at the original image resolution.

**Key Highlights:**
- **Full-Resolution Inference:** Supports direct registration of 512 × 512 images.
- **Efficient Training:** Maintains the original 64 × 64 training resolution.
- **Plug-and-Play Design:** Enables high-resolution inference without retraining the original model.
- **High Registration Accuracy:** Delivers excellent registration performance at the original image resolution.

This update significantly enhances the practical applicability of CRFT, enabling accurate high-resolution cross-modal image registration while preserving its training efficiency.

## Data Preparation

To evaluate/train CRFT, you will need to download the required datasets.

* [RoadScene](https://pan.baidu.com/s/1MLmwlUf2jdWdWSVQwwQt2w?pwd=kr84)
* [OSdataset](https://pan.baidu.com/s/1C-YckJJvCNULfpMQxy_yfA?pwd=mj8d)

You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder.

```text
datasets/
├── os_dataset/
│   ├── train/
│   │   ├── image_pair/
│   │   ├── truth_flow/
│   │   └── datum/
│   ├── test/
│   │   ├── image_pair/
│   │   ├── truth_flow/
│   │   └── datum/
│   └── val/
│       ├── image_pair/
│       ├── truth_flow/
│       └── datum/
└── RoadScene/
    ├── train/
    │   ├── image_pair/
    │   ├── truth_flow/
    │   └── datum/
    ├── test/
    │   ├── image_pair/
    │   ├── truth_flow/
    │   └── datum/
    └── val/
        ├── image_pair/
        ├── truth_flow/
        └── datum/
```

## Requirements

```shell
conda create --name crft python=3.9.7
conda activate crft

conda install pytorch=2.3.1 torchvision=0.18.1 pytorch-cuda=12.1 matplotlib tensorboard scipy opencv -c pytorch -c nvidia

pip install opencv-python==4.8.0.76
pip install numpy==1.26.4
pip install pytorch-lightning loguru joblib tqdm h5py einops
```

## Training

```shell
python train.py
```

## Models

We provide models trained on OSdataset and RoadScene, respectively.

The default checkpoint paths for evaluation are:

```text
checkpoints/
├── CRFT_OSdataset.ckpt
└── CRFT_RoadScene.ckpt
```

## Test

```shell
python test.py
```

Please ensure that the input resolution, model configuration, and checkpoint are compatible with the selected evaluation settings.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{liu2026crft,
  title={CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration},
  author={Liu, Xuecong and Ding, Mengzhu and Sun, Zixuan and Li, Zhang and Teng, Xichao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={34784--34794},
  year={2026}
}
```
