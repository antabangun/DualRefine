# DualRefine
Coming soon:

PyTorch implementation of our paper
**DualRefine: Self-Supervised Depth and Pose Estimation Through Iterative Epipolar Sampling and Refinement Toward Equilibrium**,
*Authors: Antyanta Bangunharcana, Ahmed Magd Aly, and KyungSoo Kim*
To be presented at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

## Abstract
Self-supervised multi-frame depth estimation achieves high accuracy by computing matching costs of pixel correspondences between adjacent frames, injecting geometric information into the network. These pixel-correspondence candidates are computed based on the relative pose estimates between the frames. Accurate pose predictions are essential for precise matching cost computation as they influence the epipolar geometry. Furthermore, improved depth estimates can, in turn, be used to align pose estimates. 

Inspired by traditional structure-from-motion (SfM) principles, we propose the DualRefine model, which tightly couples depth and pose estimation through a feedback loop. Our novel update pipeline uses a deep equilibrium model framework to iteratively refine depth estimates and a hidden state of feature maps by computing local matching costs based on epipolar geometry. Importantly, we used the refined depth estimates and feature maps to compute pose updates at each step. This update in the pose estimates slowly alters the epipolar geometry during the refinement process. Experimental results on the KITTI dataset demonstrate competitive depth prediction and odometry prediction performance surpassing published self-supervised baselines. 

<!-- \[[Project page](https://antabangun.github.io/projects//)\] \[[Paper](https://arxiv.org/abs/)\] -->

## Contents
1. [Installation](#installation)
2. [Training](#training)
4. [Citation](#citation)

## Installation


## Training



## Citation
If you find this work useful in your research, please consider citing:
