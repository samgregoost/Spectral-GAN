# Spectral-GAN

Tensorflow implementation of [Spectral-GANs for High-Resolution 3D Point-cloud Generation](https://arxiv.org/pdf/1912.01800.pdf) (IROS 2020)


## Introduction

Point-clouds are a popular choice for vision and graphics tasks due to their accurate shape description and direct acquisition from range-scanners. This demands the ability to synthesize and reconstruct high-quality point-clouds. Current deep generative models for 3D data generally work on simplified representations (e.g.,  voxelized objects) and cannot deal with the inherent redundancy and irregularity in point-clouds.  A few recent efforts on 3D point-cloud generation offer limited resolution and their complexity grows with the increase in output resolution. In this work, we develop a principled approach to synthesize 3D point-clouds using a spectral-domain Generative Adversarial Network (GAN). This is the Tensorflow code of our paper.

## Usage

1. Set the input path to the ground truth spherical harmonics in main.py
2. Give the ground truth point cloud path in spatial_train.py as a single numpy array (num_clouds, num_points, 3)
3. Train:
> python main.py --mode=train
4. Evaluate:
> python main.py --mode=evaluate

## Citation

>@article{ramasinghe2020spectral,
  title={Spectral-GANs for High-Resolution 3D Point-cloud Generation},
  author={Ramasinghe, Sameera and Khan, Salman and Barnes, Nick and Gould, Stephen},
  journal={IEEE/RSJ Intenrational Conference on Robots and Systems (IROS)},
  year={2020}
}
