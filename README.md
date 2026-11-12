# Instrument-Splatting++

Official code implementation for [Instrument-Splatting](https://arxiv.org/abs/2503.04082), a Real2Sim framework for controllable and photorealistic surgical instrument reconstruction.

<!--### [Project Page]() -->

> [Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting](https://papers.miccai.org/miccai-2025/0453-Paper3069.html)\
> Shuojue Yang*, Zijian Wu*, Mingxuan Hong, Qian Li, Daiyun Shen, Septimiu E. Salcudean, Yueming Jin\
> Accept at MICCAI2025

## Demo

### Pose estimation & Reconstruction quality 

https://github.com/user-attachments/assets/c5a1e684-b998-489b-960c-9d7717bfd65b

### Complete pipeline

[Watch the TMI demo video](media/TMI_demo.mp4)

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

To train Instrument-Splatting with customized hyper-parameters, please make changes in `arguments/__init__.py`

### Pretrained GS model
We have uploaded our pretrained GS models to `pretrained_models`. Now we have released two options: `pretrain_3dgs_dense_param` and `pretrain_param_3dgs` with various gaussian numbers. Please choose per your `arguments/__init__.py`.

### Pose Tracking & Estimation
To estimate per-frame instrument poses, run the following example command:

```
python pose_tracking_tipNet.py --source_path data/surgpose_sample --task_name exp1
```

### Texture Learning
To learn the instrument appearances, run the following example command:
```
python texture_learning.py --source_path data/surgpose_sample --task_name exp1
```

## Evaluation

To obtain the final image quality estimation, run:

```
# TTPR is enabled by default
python infer.py --source_path data/surgpose_sample --exp_name exp1

# Disable test-time pose refinement
python infer.py --source_path data/surgpose_sample --exp_name exp1 --no-ttpr
```

The default TTPR setting performs 200 pose-refinement steps per test frame. Use `--ttpr_steps` to change this value. Results are written to `Results/<exp_name>/inference_ttpr/` or `Results/<exp_name>/inference_no_ttpr/`.

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
