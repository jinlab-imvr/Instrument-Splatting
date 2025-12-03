# Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting

Official code implementation for [Instrument-Splatting](https://arxiv.org/abs/2503.04082), a Real2Sim framework for controllable and photorealistic surgical instrument reconstruction.

<!--### [Project Page]() -->

> [Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting](https://papers.miccai.org/miccai-2025/0453-Paper3069.html)\
> Shuojue Yang*, Zijian Wu*, Mingxuan Hong, Qian Li, Daiyun Shen, Septimiu E. Salcudean, Yueming Jin\
> Accept at MICCAI2025

## Demo

### Pose estimation & Reconstruction quality 

https://github.com/user-attachments/assets/c5a1e684-b998-489b-960c-9d7717bfd65b

## Environment setup

Tested with NVIDIA RTX A5000 GPU.

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/jinlab-imvr/Instrument-Splatting.git
cd Instrument-Splatting

# If you already cloned without --recursive, initialize submodules
git submodule update --init --recursive

# Create conda environment
conda create -n instrument_splatting python=3.7 
conda activate instrument_splatting

# Install dependencies
pip install -r requirements.txt

# Install submodules
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/gaussian-rasterization

# Install ml_aspanformer dependencies
cd submodules/ml_aspanformer
pip install -r requirements.txt
pip install -e .
cd ../..
```
## Dataset
Our customized datasets follow the structure in surgpose_sample (), containing
1) `mask_all`: mask for overall instruments (not used);
2) `depth`: depth GT in the unit of mm
3) `l_masks`: part-wise mask labeled with [SAM2](https://github.com/facebookresearch/sam2)
4) `mappings.json`: color scalars to part (not used)
5) `testing_iterations.txt`: iterations for testing phase
6) `training_iterations.txt`: iterations for training phase
7) `transforms.json`: initialized 6D wrist pose for the first frame given by PnP
The data structure is as follows:

```
data
| - surgpose_sample
|   | - mask_all/
|   | - depth/
|   | - color/
|   | - l_masks/
|   | - mappings.json
|   | - testing_iterations.txt
|   | - training_iterations.txt  
|   | - transforms.json  
```
## Training

To train Instrument-Splatting with customized hyper-parameters, please make changes in `arguments/__init__.py`

### Pose Tracking & Estimation
To estimate per-frame instrument poses, run the following example command:

```
python pose_tracking.py --source_path data/surgpose_sample --exp_name exp1
```

### Texture Learning
To learn the instrument appearances, run the following example command:
```
python texture_learning.py --source_path data/surgpose_sample --exp_name exp1
```

## Evaluation

To obtain the final image quality estimation, run:

```
python infer.py --source_path data/surgpose_sample --exp_name exp1
```