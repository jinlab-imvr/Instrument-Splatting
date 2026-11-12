#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SEMANTIC_IDS = {0, 10, 20, 30}


def natural_key(path):
    return [int(value) if value.isdigit() else value.lower() for value in re.split(r"(\d+)", path.name)]


def list_images(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(directory))
    paths = sorted(
        [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if not paths:
        raise ValueError("No images found in {}".format(directory))
    return paths


def image_map(directory):
    paths = list_images(directory)
    mapping = {path.stem: path for path in paths}
    if len(mapping) != len(paths):
        raise ValueError("Duplicate image stems found in {}".format(directory))
    return mapping


def load_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def write_json(path, value):
    with open(str(path), "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def prepare_output(directory, overwrite):
    directory = Path(directory).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "Output directory is not empty; pass --overwrite to replace it: {}".format(
                    directory
                )
            )
        shutil.rmtree(str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError("OpenCV could not read {}".format(path))
    return image


def resize_image(image, width, height, interpolation):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=interpolation)


def parse_matrix(value, name, rows, columns):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (rows, columns):
        raise ValueError("{} must have shape {}x{}".format(name, rows, columns))
    return matrix


def parse_distortion(value, name):
    distortion = np.asarray(value, dtype=np.float64).reshape(-1)
    if distortion.size not in (4, 5, 8, 12, 14):
        raise ValueError("{} must contain 4, 5, 8, 12, or 14 values".format(name))
    return distortion


def command_extract_video(args):
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.stop_frame is not None and args.stop_frame <= args.start_frame:
        raise ValueError("--stop-frame must be greater than --start-frame")
    if args.stride < 1:
        raise ValueError("--stride must be positive")

    output_dir = prepare_output(args.output_dir, args.overwrite)
    capture = cv2.VideoCapture(str(Path(args.video).expanduser().resolve()))
    if not capture.isOpened():
        raise ValueError("Could not open video: {}".format(args.video))

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stop_frame = total_frames if args.stop_frame is None else min(args.stop_frame, total_frames)
    output_width = source_width if args.width is None else args.width
    output_height = source_height if args.height is None else args.height
    if output_width < 1 or output_height < 1:
        raise ValueError("Output width and height must be positive")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    source_frame = args.start_frame
    output_index = 0
    manifest_frames = []
    while source_frame < stop_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (source_frame - args.start_frame) % args.stride == 0:
            frame = resize_image(frame, output_width, output_height, cv2.INTER_LINEAR)
            name = "{}{:0{}d}.png".format(args.prefix, output_index, args.digits)
            destination = output_dir / name
            if not cv2.imwrite(str(destination), frame):
                raise IOError("Could not write {}".format(destination))
            manifest_frames.append(
                {"output_file": name, "source_frame_id": source_frame}
            )
            output_index += 1
        source_frame += 1
    capture.release()

    if not manifest_frames:
        raise RuntimeError("No frames were extracted")
    manifest = {
        "source_video": str(Path(args.video).expanduser().resolve()),
        "source_frame_count": total_frames,
        "source_resolution": [source_width, source_height],
        "output_resolution": [output_width, output_height],
        "start_frame": args.start_frame,
        "stop_frame_exclusive": stop_frame,
        "stride": args.stride,
        "frames": manifest_frames,
    }
    write_json(output_dir / "frame_manifest.json", manifest)
    print("Extracted {} frames to {}".format(len(manifest_frames), output_dir))


def calibration_value(calibration, names):
    for name in names:
        if name in calibration:
            return calibration[name]
    raise KeyError("Calibration is missing one of: {}".format(", ".join(names)))


def command_rectify_stereo(args):
    left_images = image_map(args.left_dir)
    right_images = image_map(args.right_dir)
    if set(left_images) != set(right_images):
        missing_left = sorted(set(right_images) - set(left_images))
        missing_right = sorted(set(left_images) - set(right_images))
        raise ValueError(
            "Stereo filenames do not match. Missing left: {}; missing right: {}".format(
                missing_left[:10], missing_right[:10]
            )
        )

    output_dir = prepare_output(args.output_dir, args.overwrite)
    output_left = output_dir / "left"
    output_right = output_dir / "right"
    output_left.mkdir()
    output_right.mkdir()

    calibration = load_json(Path(args.calibration))
    left_k = parse_matrix(
        calibration_value(calibration, ("K_left", "camera_matrix_left")),
        "K_left",
        3,
        3,
    )
    right_k = parse_matrix(
        calibration_value(calibration, ("K_right", "camera_matrix_right")),
        "K_right",
        3,
        3,
    )
    left_d = parse_distortion(
        calibration_value(calibration, ("D_left", "distortion_left")), "D_left"
    )
    right_d = parse_distortion(
        calibration_value(calibration, ("D_right", "distortion_right")), "D_right"
    )
    rotation = parse_matrix(calibration_value(calibration, ("R", "rotation")), "R", 3, 3)
    translation = np.asarray(
        calibration_value(calibration, ("T", "translation")), dtype=np.float64
    ).reshape(-1)
    if translation.size != 3:
        raise ValueError("T must contain three values")

    first_name = sorted(left_images, key=lambda value: natural_key(left_images[value]))[0]
    first_left = read_image(left_images[first_name], cv2.IMREAD_COLOR)
    first_right = read_image(right_images[first_name], cv2.IMREAD_COLOR)
    if first_left.shape[:2] != first_right.shape[:2]:
        raise ValueError("Left and right images have different resolutions")
    source_height, source_width = first_left.shape[:2]
    output_width = source_width if args.width is None else args.width
    output_height = source_height if args.height is None else args.height

    rectify = cv2.stereoRectify(
        left_k,
        left_d,
        right_k,
        right_d,
        (source_width, source_height),
        rotation,
        translation,
        alpha=args.alpha,
        newImageSize=(output_width, output_height),
    )
    left_r, right_r, left_p, right_p, disparity_q = rectify[:5]
    left_map = cv2.initUndistortRectifyMap(
        left_k,
        left_d,
        left_r,
        left_p[:, :3],
        (output_width, output_height),
        cv2.CV_32FC1,
    )
    right_map = cv2.initUndistortRectifyMap(
        right_k,
        right_d,
        right_r,
        right_p[:, :3],
        (output_width, output_height),
        cv2.CV_32FC1,
    )

    ordered_names = sorted(left_images, key=lambda value: natural_key(left_images[value]))
    for name in ordered_names:
        left = read_image(left_images[name], cv2.IMREAD_COLOR)
        right = read_image(right_images[name], cv2.IMREAD_COLOR)
        if left.shape[:2] != (source_height, source_width):
            raise ValueError("Inconsistent left resolution: {}".format(left_images[name]))
        if right.shape[:2] != (source_height, source_width):
            raise ValueError("Inconsistent right resolution: {}".format(right_images[name]))
        rectified_left = cv2.remap(left, left_map[0], left_map[1], cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right, right_map[0], right_map[1], cv2.INTER_LINEAR)
        cv2.imwrite(str(output_left / (name + ".png")), rectified_left)
        cv2.imwrite(str(output_right / (name + ".png")), rectified_right)

    metadata = {
        "image_width": output_width,
        "image_height": output_height,
        "intrinsic_matrix": left_p[:, :3].tolist(),
        "right_intrinsic_matrix": right_p[:, :3].tolist(),
        "left_projection_matrix": left_p.tolist(),
        "right_projection_matrix": right_p.tolist(),
        "disparity_to_depth_matrix": disparity_q.tolist(),
        "frame_count": len(ordered_names),
    }
    write_json(output_dir / "rectification.json", metadata)
    print("Rectified {} stereo pairs to {}".format(len(ordered_names), output_dir))


def binary_mask(path, width, height):
    mask = read_image(path, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = np.max(mask, axis=2)
    mask = resize_image(mask, width, height, cv2.INTER_NEAREST)
    return mask > 0


def command_combine_masks(args):
    shaft = image_map(args.shaft_dir)
    wrist = image_map(args.wrist_dir)
    gripper = image_map(args.gripper_dir)
    if not (set(shaft) == set(wrist) == set(gripper)):
        raise ValueError("Shaft, wrist, and gripper mask filenames must match")

    output_dir = prepare_output(args.output_dir, args.overwrite)
    ordered_names = sorted(shaft, key=lambda value: natural_key(shaft[value]))
    for name in ordered_names:
        reference = read_image(shaft[name], cv2.IMREAD_UNCHANGED)
        height = reference.shape[0] if args.height is None else args.height
        width = reference.shape[1] if args.width is None else args.width
        semantic = np.zeros((height, width), dtype=np.uint8)
        semantic[binary_mask(shaft[name], width, height)] = 10
        semantic[binary_mask(wrist[name], width, height)] = 20
        semantic[binary_mask(gripper[name], width, height)] = 30
        cv2.imwrite(str(output_dir / (name + ".png")), semantic)
    print("Combined {} semantic masks in {}".format(len(ordered_names), output_dir))


def camera_configuration(path):
    config = load_json(Path(path))
    intrinsic = parse_matrix(config["intrinsic_matrix"], "intrinsic_matrix", 3, 3)
    if "initial_transform" in config:
        initial_transform = config["initial_transform"]
    elif "transform_matrix" in config:
        initial_transform = config["transform_matrix"]
    elif config.get("frames"):
        initial_transform = config["frames"][0]["transform_matrix"]
    else:
        raise KeyError(
            "Camera JSON must contain initial_transform, transform_matrix, or frames[0]"
        )
    initial_transform = parse_matrix(
        initial_transform, "initial_transform", 4, 4
    )
    return config, intrinsic, initial_transform


def selected_test_ids(args, frame_count):
    if args.test_ids is not None:
        test_ids = args.test_ids
    elif args.test_ids_file is not None:
        text = Path(args.test_ids_file).read_text()
        test_ids = [int(value) for value in re.findall(r"\d+", text)]
    else:
        test_every = 8 if args.test_every is None else args.test_every
        if test_every < 2:
            raise ValueError("--test-every must be at least 2")
        test_ids = list(range(test_every - 1, frame_count, test_every))
    test_ids = sorted(set(test_ids))
    invalid = [frame_id for frame_id in test_ids if frame_id < 0 or frame_id >= frame_count]
    if invalid:
        raise ValueError("Test frame IDs are outside the dataset: {}".format(invalid))
    if not test_ids:
        raise ValueError("At least one test frame is required")
    if len(test_ids) == frame_count:
        raise ValueError("At least one training frame is required")
    return test_ids


def write_split(path, frame_ids):
    with open(str(path), "w") as handle:
        for frame_id in frame_ids:
            handle.write("{}\n".format(frame_id))


def check_source_outside_output(source, output):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    try:
        common_path = os.path.commonpath((str(source), str(output)))
    except ValueError:
        return
    if common_path == str(output):
        raise ValueError("Input directory cannot be inside the output directory: {}".format(source))


def validate_semantic_mask(mask, path):
    if mask.ndim != 2:
        raise ValueError("Semantic mask must be a single-channel label image: {}".format(path))
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset(SEMANTIC_IDS):
        raise ValueError(
            "Semantic mask {} contains unexpected IDs: {}".format(
                path, sorted(values - SEMANTIC_IDS)
            )
        )


def command_assemble(args):
    check_source_outside_output(args.color_dir, args.output_dir)
    check_source_outside_output(args.semantic_mask_dir, args.output_dir)
    if args.depth_dir is not None:
        check_source_outside_output(args.depth_dir, args.output_dir)

    color_paths = list_images(args.color_dir)
    semantic_paths = image_map(args.semantic_mask_dir)
    depth_paths = image_map(args.depth_dir) if args.depth_dir is not None else None
    for path in color_paths:
        if path.stem not in semantic_paths:
            raise ValueError("Missing semantic mask for {}".format(path.name))
        if depth_paths is not None and path.stem not in depth_paths:
            raise ValueError("Missing depth map for {}".format(path.name))

    first_color = read_image(color_paths[0], cv2.IMREAD_COLOR)
    source_height, source_width = first_color.shape[:2]
    output_width = source_width if args.width is None else args.width
    output_height = source_height if args.height is None else args.height
    if output_width < 1 or output_height < 1:
        raise ValueError("Output width and height must be positive")

    camera_config, intrinsic, initial_transform = camera_configuration(args.camera_json)
    calibration_width = int(camera_config.get("image_width", source_width))
    calibration_height = int(camera_config.get("image_height", source_height))
    intrinsic = intrinsic.copy()
    intrinsic[0, :] *= float(output_width) / float(calibration_width)
    intrinsic[1, :] *= float(output_height) / float(calibration_height)

    output_dir = prepare_output(args.output_dir, args.overwrite)
    color_output = output_dir / "color"
    semantic_output = output_dir / "l_mask"
    overall_output = output_dir / "mask_all"
    color_output.mkdir()
    semantic_output.mkdir()
    overall_output.mkdir()
    depth_output = None
    if depth_paths is not None:
        depth_output = output_dir / "depth"
        depth_output.mkdir()

    frames = []
    manifest_frames = []
    for frame_id, color_path in enumerate(color_paths):
        output_name = "frame{:03d}.png".format(frame_id)
        color = read_image(color_path, cv2.IMREAD_COLOR)
        color = resize_image(color, output_width, output_height, cv2.INTER_LINEAR)
        semantic_path = semantic_paths[color_path.stem]
        semantic = read_image(semantic_path, cv2.IMREAD_UNCHANGED)
        semantic = resize_image(
            semantic, output_width, output_height, cv2.INTER_NEAREST
        )
        validate_semantic_mask(semantic, semantic_path)
        overall = np.where(semantic > 0, 255, 0).astype(np.uint8)

        cv2.imwrite(str(color_output / output_name), color)
        cv2.imwrite(str(semantic_output / output_name), semantic)
        cv2.imwrite(str(overall_output / output_name), overall)
        if depth_output is not None:
            depth = read_image(depth_paths[color_path.stem], cv2.IMREAD_UNCHANGED)
            depth = resize_image(depth, output_width, output_height, cv2.INTER_NEAREST)
            cv2.imwrite(str(depth_output / output_name), depth)

        frames.append(
            {
                "file_path": "color/{}".format(Path(output_name).stem),
                "transform_matrix": initial_transform.tolist(),
            }
        )
        manifest_frames.append(
            {
                "frame_id": frame_id,
                "output_file": output_name,
                "source_color": str(color_path),
                "source_semantic_mask": str(semantic_path),
                "source_depth": str(depth_paths[color_path.stem])
                if depth_paths is not None
                else None,
            }
        )

    test_ids = selected_test_ids(args, len(frames))
    test_set = set(test_ids)
    training_ids = [frame_id for frame_id in range(len(frames)) if frame_id not in test_set]
    transforms = {
        "camera_angle_x": 2.0 * math.atan(output_width / (2.0 * intrinsic[0, 0])),
        "intrinsic_matrix": intrinsic.tolist(),
        "frames": frames,
    }
    write_json(output_dir / "transforms.json", transforms)
    write_json(
        output_dir / "mappings.json",
        {"Background": 0, "Shaft": 10, "Wrist": 20, "Claspers": 30},
    )
    write_split(output_dir / "training_iterations.txt", training_ids)
    write_split(output_dir / "testing_iterations.txt", test_ids)
    write_json(
        output_dir / "processing_manifest.json",
        {
            "output_resolution": [output_width, output_height],
            "frame_count": len(frames),
            "training_frame_ids": training_ids,
            "testing_frame_ids": test_ids,
            "frames": manifest_frames,
        },
    )
    validate_dataset(output_dir)
    print("Prepared {} frames in {}".format(len(frames), output_dir))


def read_split(path):
    with open(str(path), "r") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def validate_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    required_files = (
        "transforms.json",
        "mappings.json",
        "training_iterations.txt",
        "testing_iterations.txt",
    )
    for name in required_files:
        if not (dataset_dir / name).is_file():
            raise FileNotFoundError("Missing dataset file: {}".format(dataset_dir / name))
    for name in ("color", "l_mask", "mask_all"):
        if not (dataset_dir / name).is_dir():
            raise FileNotFoundError("Missing dataset directory: {}".format(dataset_dir / name))

    transforms = load_json(dataset_dir / "transforms.json")
    intrinsic = parse_matrix(transforms["intrinsic_matrix"], "intrinsic_matrix", 3, 3)
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError("transforms.json contains no frames")

    expected_size = None
    for frame_id, frame in enumerate(frames):
        relative = frame["file_path"]
        color_path = dataset_dir / (relative + ".png")
        semantic_path = dataset_dir / "l_mask" / color_path.name
        overall_path = dataset_dir / "mask_all" / color_path.name
        for path in (color_path, semantic_path, overall_path):
            if not path.is_file():
                raise FileNotFoundError("Missing frame {}: {}".format(frame_id, path))
        color = read_image(color_path, cv2.IMREAD_COLOR)
        semantic = read_image(semantic_path, cv2.IMREAD_UNCHANGED)
        overall = read_image(overall_path, cv2.IMREAD_UNCHANGED)
        validate_semantic_mask(semantic, semantic_path)
        size = color.shape[:2]
        if semantic.shape[:2] != size or overall.shape[:2] != size:
            raise ValueError("Frame modalities have different sizes at frame {}".format(frame_id))
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError("Color frames have inconsistent sizes")
        parse_matrix(frame["transform_matrix"], "transform_matrix", 4, 4)

    training_ids = read_split(dataset_dir / "training_iterations.txt")
    testing_ids = read_split(dataset_dir / "testing_iterations.txt")
    if not training_ids:
        training_ids = sorted(set(range(len(frames))) - set(testing_ids))
    if set(training_ids) & set(testing_ids):
        raise ValueError("Training and testing splits overlap")
    expected_ids = set(range(len(frames)))
    if set(training_ids) | set(testing_ids) != expected_ids:
        raise ValueError("Training and testing splits do not cover every frame")
    print(
        "Validated {} frames ({} training, {} testing) in {}".format(
            len(frames), len(training_ids), len(testing_ids), dataset_dir
        )
    )


def command_validate(args):
    validate_dataset(args.dataset_dir)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Instrument-Splatting++"
    )
    subparsers = parser.add_subparsers(dest="command")

    extract = subparsers.add_parser("extract-video", help="Extract and resize video frames")
    extract.add_argument("--video", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--start-frame", type=int, default=0)
    extract.add_argument("--stop-frame", type=int, default=None)
    extract.add_argument("--stride", type=int, default=1)
    extract.add_argument("--width", type=int, default=None)
    extract.add_argument("--height", type=int, default=None)
    extract.add_argument("--prefix", default="frame")
    extract.add_argument("--digits", type=int, default=3)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(function=command_extract_video)

    rectify = subparsers.add_parser("rectify-stereo", help="Rectify stereo frame pairs")
    rectify.add_argument("--left-dir", required=True)
    rectify.add_argument("--right-dir", required=True)
    rectify.add_argument("--calibration", required=True)
    rectify.add_argument("--output-dir", required=True)
    rectify.add_argument("--width", type=int, default=None)
    rectify.add_argument("--height", type=int, default=None)
    rectify.add_argument("--alpha", type=float, default=0.0)
    rectify.add_argument("--overwrite", action="store_true")
    rectify.set_defaults(function=command_rectify_stereo)

    combine = subparsers.add_parser(
        "combine-masks", help="Combine binary part masks into one semantic mask"
    )
    combine.add_argument("--shaft-dir", required=True)
    combine.add_argument("--wrist-dir", required=True)
    combine.add_argument("--gripper-dir", required=True)
    combine.add_argument("--output-dir", required=True)
    combine.add_argument("--width", type=int, default=None)
    combine.add_argument("--height", type=int, default=None)
    combine.add_argument("--overwrite", action="store_true")
    combine.set_defaults(function=command_combine_masks)

    assemble = subparsers.add_parser(
        "assemble", help="Assemble the final Instrument-Splatting++ dataset"
    )
    assemble.add_argument("--color-dir", required=True)
    assemble.add_argument("--semantic-mask-dir", required=True)
    assemble.add_argument("--depth-dir", default=None)
    assemble.add_argument("--camera-json", required=True)
    assemble.add_argument("--output-dir", required=True)
    assemble.add_argument("--width", type=int, default=None)
    assemble.add_argument("--height", type=int, default=None)
    split_group = assemble.add_mutually_exclusive_group()
    split_group.add_argument("--test-ids", nargs="+", type=int, default=None)
    split_group.add_argument("--test-ids-file", default=None)
    split_group.add_argument("--test-every", type=int, default=None)
    assemble.add_argument("--overwrite", action="store_true")
    assemble.set_defaults(function=command_assemble)

    validate = subparsers.add_parser("validate", help="Validate a prepared dataset")
    validate.add_argument("--dataset-dir", required=True)
    validate.set_defaults(function=command_validate)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "function"):
        parser.print_help()
        return 2
    try:
        args.function(args)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
