# Data Preprocessing

This folder provides the conversion pipeline for the Instrument-Splatting++ In-house dataset. The public release contains the first 100 sampled frames of two clips. Their original source-video frame IDs are listed in `frame_sampling_and_resize.txt`.

## In-house dataset

- Processed data: [In-house dataset](https://drive.google.com/file/d/1qAv-YpouDOFbf7a4epeeuXkx5GW5sCFo/view?usp=sharing)
- Sampling and resize record: [`frame_sampling_and_resize.txt`](frame_sampling_and_resize.txt)

## 1. Extract frames

The following example retains every second frame, including the first selected frame, and resizes each image without cropping or padding:

```bash
python data_preprocessing/process_dataset.py extract-video \
  --video /path/to/left_video.mp4 \
  --output-dir work/left_raw \
  --start-frame 3312 \
  --stop-frame 3512 \
  --stride 2 \
  --width 640 \
  --height 352
```

`--stop-frame` is exclusive, so this command extracts the first clip's 100 source frames from 3312 through 3510. For the second clip, use `--start-frame 966 --stop-frame 1166`. Apply the same selection to each synchronized right video.

## 2. Rectify stereo images

Stereo rectification is needed before stereo depth estimation. Supply an OpenCV-style JSON file containing `K_left`, `D_left`, `K_right`, `D_right`, `R`, and `T`:

```bash
python data_preprocessing/process_dataset.py rectify-stereo \
  --left-dir work/left_raw \
  --right-dir work/right_raw \
  --calibration /path/to/stereo_calibration.json \
  --output-dir work/rectified
```

This command writes rectified `left/` and `right/` images and a `rectification.json` file containing the rectified camera matrix. Skip this step when the source images are already rectified.

## 3. Estimate stereo depth

Depth maps are estimated from the rectified stereo pairs with [MonSter](https://github.com/Junda24/MonSter). Install MonSter and download its pretrained weights by following the upstream instructions, then run the included wrapper:

```bash
git clone https://github.com/Junda24/MonSter.git third_party/MonSter

python data_preprocessing/estimate_depth_monster.py \
  --monster-root third_party/MonSter \
  --restore-ckpt third_party/MonSter/pretrained/mix_all.pth \
  --left-dir work/rectified/left \
  --right-dir work/rectified/right \
  --rectification-json work/rectified/rectification.json \
  --output-dir work/depth
```

The wrapper saves float32 depth arrays and 16-bit PNG depth maps. Depth has the same linear unit as the stereo translation used during rectification; use millimetres for `T` to produce the millimetre depth maps expected by this dataset. MonSter uses a separate Python 3.8 environment and its own dependencies.

## 4. Generate part masks

Annotate the shaft, wrist, and gripper with the [SAM2 Annotation Tool](https://github.com/lala-sean/SAM2_Annotation_Tool), export one binary-mask directory per part, and combine them into the scalar semantic labels expected by this repository:

```bash
python data_preprocessing/process_dataset.py combine-masks \
  --shaft-dir work/masks_shaft \
  --wrist-dir work/masks_wrist \
  --gripper-dir work/masks_gripper \
  --output-dir work/l_mask
```

Overlapping pixels use the priority `gripper > wrist > shaft`. The output labels are background `0`, shaft `10`, wrist `20`, and gripper `30`.

## 5. Assemble a dataset

Create a camera JSON following `camera.example.json` and replace its example values with the calibration and first-frame pose for your sequence. `initial_transform` is the first-frame wrist-to-camera initialization. The intrinsic matrix is automatically rescaled when `--width` and `--height` differ from `image_width` and `image_height` in the camera JSON.

```bash
python data_preprocessing/process_dataset.py assemble \
  --color-dir work/rectified/left \
  --semantic-mask-dir work/l_mask \
  --depth-dir work/depth \
  --camera-json data_preprocessing/camera.example.json \
  --output-dir data/in_house_dataset \
  --test-every 8
```

Use `--test-ids 5 27 33` or `--test-ids-file split.txt` to reproduce a fixed split. When no split option is provided, every eighth frame is assigned to testing. The depth directory is optional because the current tracking, RTL, and inference entry points do not consume depth.

The assembled dataset contains:

```text
in_house_dataset/
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

## 6. Validate

```bash
python data_preprocessing/process_dataset.py validate --dataset-dir data/in_house_dataset
```

Validation checks image availability and resolution, semantic IDs, camera matrices, initial transforms, and disjoint train/test splits that cover every frame.
