# Dual Diffusion

## Code for ["Dynamical Dual-Output Diffusion Models"](https://arxiv.org/pdf/2203.04304.pdf), CVPR 2022.


## Requirements
This repository uses
* python 3.8.12
* pyyaml, tqdm
* numpy, matplotlib
* omegaconf 2.1.1
* torch 1.7.1, torchvision 0.8.2
* pytorch-lightning 1.5.2
* weights & biases (for monitoring) 0.12.7

## Usage
This repo contains config file and instruction for training and evaluating on cifar10.

To train the model:

`> python src/train.py --config config/cifar10_train.yaml`

To plot losses:

`> python src/plot_losses.py --config config/cifar10_train.yaml --model_paths <MODEL_PATH>`

Where \<MODEL_PATHS\> can be one or more paths to pretrained models.

To compute FID:

`> python src/compute_fid.py --model_path <MODEL_PATH> --num_timesteps <N>`  

Where \<MODEL_PATH\> is a path to one pretrained model and \<N\> is the number of denoising steps.

## Citation
We thank you for showing interest in our work. 
If our work was beneficial for you, please consider citing us using:

```
@inproceedings{benny2022dynamic,
  title={Dynamic Dual-Output Diffusion Models},
  author={Benny, Yaniv and Wolf, Lior},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11482--11491},
  year={2022}
}
```