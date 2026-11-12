# Instrument-Splatting++

Official code implementation for [Instrument-Splatting](https://arxiv.org/abs/2503.04082), a Real2Sim framework for controllable and photorealistic surgical instrument reconstruction.

<!--### [Project Page]() -->

> [Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting](https://papers.miccai.org/miccai-2025/0453-Paper3069.html)\
> Shuojue Yang*, Zijian Wu*, Mingxuan Hong, Qian Li, Daiyun Shen, Septimiu E. Salcudean, Yueming Jin\
> Accept at MICCAI2025

## Demo

### MIA_demp

![MIA_demp](media/MIA_demp_preview.webp)

[Open the full-resolution video](media/MIA_demp.mp4)

## Environment setup

Tested with NVIDIA RTX A5000 GPU.

```bash
# Clone the Instrument-Splatting++ branch
git clone --branch 'instrument-splatting++' https://github.com/jinlab-imvr/Instrument-Splatting.git
cd Instrument-Splatting

# Create conda environment
conda create -n instrument_splatting python=3.7 
conda activate instrument_splatting

# Install dependencies
pip install -r requirements.txt

# Install submodules
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn

# Install ml_aspanformer dependencies
cd submodules/ml_aspanformer
pip install -r requirements.txt

cd ../..
```

All runtime model assets are included in the repository: the four instrument CAD models, two sets of initialized GS models, the Gripper-Tip Net checkpoint, ASpanFormer weights, and the AlexNet/LPIPS evaluation weights. The scripts do not download model weights at runtime.

## Dataset
Our customized datasets follow the structure in [surgpose_sample](https://drive.google.com/file/d/1qAv-YpouDOFbf7a4epeeuXkx5GW5sCFo/view?usp=sharing). The complete preparation and validation pipeline is documented in [`data_processing/README.md`](data_processing/README.md). Each dataset contains:
1) `color`: left RGB images after stereo rectification;
2) `mask_all`: mask for overall instruments (not used);
3) `depth`: optional depth GT with unit of mm, estimated by stereo matching with [MonSter](https://github.com/Junda24/MonSter);
4) `l_mask`: part-wise mask labeled with [SAM2](https://github.com/facebookresearch/sam2);
5) `mappings.json`: color scalars to part (not used);
6) `testing_iterations.txt`: iterations for testing phase;
7) `training_iterations.txt`: iterations for training phase;
8) `transforms.json`: initialized 6D wrist pose for the first frame given by PnP;

The data structure is as follows:

```
data
| - surgpose_sample
|   | - mask_all/
|   | - depth/
|   | - color/
|   | - l_mask/
|   | - mappings.json
|   | - testing_iterations.txt
|   | - training_iterations.txt  
|   | - transforms.json  
```
## Training

Training consists of two stages: pose tracking followed by reconstruction texture learning (RTL). Use the same experiment name and pretrained GS option for both stages.

### Pretrained GS Models

The required initialized GS models are included in `pretrained_models/`. Two variants are available: `pretrain_param_3dgs` and `pretrain_3dgs_dense_param`. Select one with `--pretrain_params`.

### 1. Pose Tracking and Estimation

Estimate the per-frame instrument poses:

```bash
python pose_tracking_tipNet.py \
    --source_path data/surgpose_sample \
    --task_name exp1 \
    --pretrain_params pretrain_3dgs_dense_param
```

This stage writes the tracked poses to `Results/exp1/memory_pool.pth` and saves tracking visualizations under `Results/exp1/tracking_visualization/`. Use `--tracking_iterations` to change the number of optimization steps per frame and `--max_frames` for a short run.

### 2. Reconstruction Texture Learning

Run RTL with the tracking result from the first stage:

```bash
python texture_learning.py \
    --source_path data/surgpose_sample \
    --task_name exp1 \
    --pretrain_params pretrain_3dgs_dense_param
```

By default, this command reads `Results/exp1/memory_pool.pth`. It uses the frames listed in `training_iterations.txt`; when that file is empty, it uses all tracked frames not listed in `testing_iterations.txt`. The four trained instrument-part models and the updated `memory_pool.pth` are saved in `Results/exp1/`. Use `--texture_learning_iterations` to change the number of RTL steps, or `--memory_pool <path>` to load a tracking result from another location.

## Inference and Evaluation

Inference loads the trained part models and `memory_pool.pth` from `Results/<exp_name>/`, then evaluates the frames listed in `testing_iterations.txt`.

```bash
# TTPR is enabled by default
python infer.py --source_path data/surgpose_sample --exp_name exp1

# Use a custom number of TTPR steps
python infer.py --source_path data/surgpose_sample --exp_name exp1 --ttpr_steps 100

# Disable test-time pose refinement
python infer.py --source_path data/surgpose_sample --exp_name exp1 --no-ttpr
```

TTPR performs 200 pose-refinement steps per test frame by default. Use `--max_test_frames <N>` for a short inference run.

Results are written to `Results/<exp_name>/inference_ttpr/` or `Results/<exp_name>/inference_no_ttpr/` and include:

- `renders_rgb/`: final RGB renderings;
- `renders_rgb_before_ttpr/`: renderings before refinement when TTPR is enabled;
- `renders_depth/`: depth arrays and preview images;
- `image_metrics.json`: per-frame PSNR, SSIM, and LPIPS;
- `summary.json`: aggregate metrics;
- `refined_pose_values.pth`: refined test poses when TTPR is enabled.

## Citation

If you find this code useful for your research, please use the following BibTeX entries:

```
@InProceedings{Yang_InstrumentSplatting_MICCAI2025,
        author = { Yang, Shuojue AND Wu, Zijian AND Hong, Mingxuan AND Li, Qian AND Shen, Daiyun AND Salcudean, Septimiu E. AND Jin, Yueming},
        title = { { Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15962},
        month = {September},
        page = {301 -- 311}
}

```

### Questions

For further question about the code or paper, welcome to create an issue, or contact 's.yang@u.nus.edu'
