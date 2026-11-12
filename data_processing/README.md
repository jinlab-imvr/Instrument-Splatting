# Data Processing

This folder provides the complete conversion pipeline for custom Instrument-Splatting++ datasets. The processing tool can extract video clips, rectify stereo images, combine binary part masks, assemble the required dataset structure, and validate the result.

## Processed datasets

- Processed RARP data: **TODO**
- Processed in-house data: **TODO**
- Small format example: [surgpose_sample](https://drive.google.com/file/d/1qAv-YpouDOFbf7a4epeeuXkx5GW5sCFo/view?usp=sharing)

## 1. Extract frames

The following example retains every second frame, including the first selected frame, and resizes each image without cropping or padding:

```bash
python data_processing/process_dataset.py extract-video \
  --video /path/to/left_video.mp4 \
  --output-dir work/left_raw \
  --start-frame 3312 \
  --stop-frame 3548 \
  --stride 2 \
  --width 640 \
  --height 352
```

The exact frame selections used for the released RARP clips are recorded in `manifests/sar_rarp50_frame_sampling_and_resize.txt`.

## 2. Rectify stereo images

Stereo rectification is needed before stereo depth estimation. Supply an OpenCV-style JSON file containing `K_left`, `D_left`, `K_right`, `D_right`, `R`, and `T`:

```bash
python data_processing/process_dataset.py rectify-stereo \
  --left-dir work/left_raw \
  --right-dir work/right_raw \
  --calibration /path/to/stereo_calibration.json \
  --output-dir work/rectified
```

This command writes rectified `left/` and `right/` images and a `rectification.json` file containing the rectified camera matrix. Skip this step when the source images are already rectified.

## 3. Generate depth and part masks

Run [MonSter](https://github.com/Junda24/MonSter) on the rectified stereo pairs to obtain optional depth maps. Generate binary shaft, wrist, and gripper masks with [SAM2](https://github.com/facebookresearch/sam2), then combine them into the scalar semantic labels expected by this repository:

```bash
python data_processing/process_dataset.py combine-masks \
  --shaft-dir work/masks_shaft \
  --wrist-dir work/masks_wrist \
  --gripper-dir work/masks_gripper \
  --output-dir work/l_mask
```

Overlapping pixels use the priority `gripper > wrist > shaft`. The output labels are background `0`, shaft `10`, wrist `20`, and gripper `30`.

## 4. Assemble a dataset

Create a camera JSON following `camera.example.json` and replace its example values with the calibration and first-frame pose for your sequence. `initial_transform` is the first-frame wrist-to-camera initialization. The intrinsic matrix is automatically rescaled when `--width` and `--height` differ from `image_width` and `image_height` in the camera JSON.

```bash
python data_processing/process_dataset.py assemble \
  --color-dir work/rectified/left \
  --semantic-mask-dir work/l_mask \
  --depth-dir work/depth \
  --camera-json data_processing/camera.example.json \
  --output-dir data/my_sequence \
  --test-every 8
```

Use `--test-ids 5 27 33` or `--test-ids-file split.txt` to reproduce a fixed split. When no split option is provided, every eighth frame is assigned to testing. The depth directory is optional because the current tracking, RTL, and inference entry points do not consume depth.

The assembled dataset contains:

```text
my_sequence/
├── color/
├── depth/                    # optional
├── l_mask/
├── mask_all/
├── mappings.json
├── processing_manifest.json
├── testing_iterations.txt
├── training_iterations.txt
└── transforms.json
```

## 5. Validate

```bash
python data_processing/process_dataset.py validate --dataset-dir data/my_sequence
```

Validation checks image availability and resolution, semantic IDs, camera matrices, initial transforms, and disjoint train/test splits that cover every frame.
