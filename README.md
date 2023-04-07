<h1 align="center"> DualRefine: Self-Supervised Depth and Pose Estimation Through Iterative Epipolar Sampling and Refinement Toward Equilibrium
</h1>


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Conference](https://img.shields.io/badge/CVPR-2023-blue)](https://cvpr2023.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b)](https://arxiv.org/)

This is the official PyTorch implementation of the paper:

**DualRefine: Self-Supervised Depth and Pose Estimation Through Iterative Epipolar Sampling and Refinement Toward Equilibrium**

*Authors: [Antyanta Bangunharcana](https://antabangun.github.io/)<sup>1</sup>, Ahmed Magd Aly<sup>2</sup>, and KyungSoo Kim<sup>1</sup>*  
<sup>1</sup>MSC Lab, <sup>2</sup>VDC Lab, Korea Advanced Institute of Science and Technology (KAIST)  
To be presented at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.


<!-- ## Abstract

Self-supervised multi-frame depth estimation achieves high accuracy by computing matching costs of pixel correspondences between adjacent frames, injecting geometric information into the network. These pixel-correspondence candidates are computed based on the relative pose estimates between the frames. Accurate pose predictions are essential for precise matching cost computation as they influence the epipolar geometry. Furthermore, improved depth estimates can, in turn, be used to align pose estimates.

Inspired by traditional structure-from-motion (SfM) principles, we propose the DualRefine model, which tightly couples depth and pose estimation through a feedback loop. Our novel update pipeline uses a deep equilibrium model framework to iteratively refine depth estimates and a hidden state of feature maps by computing local matching costs based on epipolar geometry. Importantly, we used the refined depth estimates and feature maps to compute pose updates at each step. This update in the pose estimates slowly alters the epipolar geometry during the refinement process. Experimental results on the KITTI dataset demonstrate competitive depth prediction and odometry prediction performance surpassing published self-supervised baselines. -->

<h2 align="center">

[Project page](https://antabangun.github.io/projects/DualRefine/) | [Paper](https://arxiv.org/abs/)

</h2>

<p align="center">
  <img src="https://dl.dropboxusercontent.com/s/6uq2ppv2o6uwsn5/DualRefine.jpg" alt="Overall pipeline" />
</p>

**Overview:**
This repository contains the code for the paper DualRefine, a self-supervised depth and pose estimation model. It refines depth and pose estimates by computing matching costs based on epipolar geometry and deep equilibrium model.

<!-- **Contributions**
* We propose a novel self-supervised depth and pose estimation model, DualRefine, which iteratively refines depth and pose estimates by computing matching costs based on epipolar geometry. -->


If you find this work useful in your research, please consider citing:  
```bibtex
@inproceedings{bangunharcana2023dualrefine,
  title={DualRefine: Self-Supervised Depth and Pose Estimation Through Iterative Epipolar Sampling and Refinement Toward Equilibrium},
  author={Bangunharcana, Antyanta and Aly, Ahmed Magd and Kim, KyungSoo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  organization={IEEE}
}
```

## Contents

1. [Getting Started](#getting-started)
2. [Pretrained Models](#pretrained-models)
3. [Data Preparation](#data-preparation)
4. [Evaluation](#evaluation)
5. [Demo](#demo)
6. [Training](#training)
7. [Acknowledgements](#acknowledgements)

## Getting Started

1. Clone this repository:

```bash
git clone https://github.com/antabangun/DualRefine.git
cd DualRefine
```

2. Create an anaconda environment and activate it (optional, but recommended):
```bash
conda env create -f environment.yml
conda activate dualrefine
```

## Pretrained Models

The following table shows the performance of the models on the KITTI dataset. You can download the pretrained models from the links below, and place them in the `weights` folder.

<table>
  <thead>
    <tr>
        <th>Model</th>
        <th>Download</th>
        <th>Resolution</th>
        <th>Abs Rel ↓</th>
        <th>Sq Rel ↓</th>
        <th>RMSE ↓</th>
        <th>RMSE log ↓</th>
        <th>a1 ↑</th>
        <th>a2 ↑</th>
        <th>a3 ↑</th>
    </tr>
    </thead>
  <tbody>
    <tr>
      <td>Monodepth2</td>
      <td></td>
      <td>192x640</td>
      <td>0.115</td>
      <td>0.903</td>
      <td>4.863</td>
      <td>0.193</td>
      <td>0.877</td>
      <td>0.958</td>
      <td>0.980</td>
    </tr>
    <tr>
      <td>Manydepth</td>
      <td></td>
      <td>192x640</td>
      <td>0.098</td>
      <td>0.770</td>
      <td>4.459</td>
      <td>0.176</td>
      <td>0.900</td>
      <td>0.965</td>
      <td>0.983</td>
    </tr>
    <!-- ... -->
    <tr style="background-color: #fafdfb;">
      <td>DualRefine_MR</td>
      <td><a href="https://www.dropbox.com/s/sp7aj09gfo9jdrh/DualRefine_MR.zip?dl=0">ckpt</a></td>
      <td>192x640</td>
      <td>0.087</td>
      <td>0.698</td>
      <td>4.234</td>
      <td>0.170</td>
      <td>0.914</td>
      <td>0.967</td>
      <td>0.983</td>
    </tr>
    <tr style="border-top: 2px solid #333;">
      <td>Manydepth (HR ResNet50)</td>
      <td></td>
      <td>320x1024</td>
      <td>0.091</td>
      <td>0.694</td>
      <td>4.245</td>
      <td>0.171</td>
      <td>0.911</td>
      <td>0.968</td>
      <td>0.983</td>
    </tr>
    <tr style="background-color: #fafdfb;">
      <td>DualRefine_HR</td>
      <td><a href="https://www.dropbox.com/s/3ak2crtw38s2v8h/DualRefine_HR.zip?dl=0">ckpt</a></td>
      <td>288x960</td>
      <td>0.087</td>
      <td>0.674</td>
      <td>4.130</td>
      <td>0.167</td>
      <td>0.915</td>
      <td>0.969</td>      
      <td>0.984</td>
    </tr>
  </tbody>
</table>


## Data Preparation
Please follow the instructions in the [Monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) repository to download and prepare the KITTI dataset.
In our experiments, we use jpeg images for training and evaluation.

## Evaluation

After downloading the dataset and the pretrained models, you can try evaluating the model by following the bash script `eval.sh`. 
The script will evaluate the model on the KITTI dataset.

<!-- ## Demo

0. Download a raw KITTI sequence from [here](http://www.cvlibs.net/datasets/kitti/raw_data.php), or simply use the sequences that are already included in the `./kitti_data` folder.

To run the demo, you can run  
```bash
python demo.py --load_weights_folder weights/DualRefine_MR
```  
  The script will run the model on the KITTI dataset.
  - You can also visualize the resulting 3D point cloud by adding  
    `--viz3d`  
    to the command line arguments. Note that this requires the [Open3D](http://www.open3d.org/) library to be installed via `pip install open3d`.   -->

## Training

1. To re-train the model as was done in the paper, you can follow the bash script `train.sh`. The script will train the model on the KITTI dataset. 
This assumes that you have downloaded and prepared the KITTI dataset as described in the previous section and placed it in the `./kitti_data` folder.
The training script will save the checkpoints in the `weights` folder.


2. Command line arguments that might be useful:
   - `--batch_size`: default 12  
     You may need to reduce the batch size if the memory of your GPU is not enough. We used an NVIDIA RTX 3090 GPU for the medium resolution model.  
   - `--mixed_precision`  
   as in the `./train.sh` script is also recommended. It helps slightly with memory usage without any loss in accuracy based on our experiments.  
   
   Other ablation arguments can be found in the `dualrefine/options.py` file.




## Acknowledgements
This repository is built upon the following repositories:
- [Manydepth](https://github.com/nianticlabs/manydepth)
- [Monodepth2](https://github.com/nianticlabs/monodepth2)
- [DIFFNet](https://github.com/brandleyzhou/DIFFNet)
- [DEQ-Flow](https://github.com/locuslab/deq-flow)
- [PixLoc](https://github.com/cvg/pixloc)  

We thank the authors for their excellent work.

-------
**Contacts:**  
If you have questions, please send an email to Antyanta Bangunharcana (antabangun@kaist.ac.kr)