#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def natural_key(path):
    return [
        int(value) if value.isdigit() else value.lower()
        for value in re.split(r"(\d+)", path.name)
    ]


def image_map(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(directory))
    paths = sorted(
        [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if not paths:
        raise ValueError("No images found in {}".format(directory))
    mapping = {path.stem: path for path in paths}
    if len(mapping) != len(paths):
        raise ValueError("Duplicate image stems found in {}".format(directory))
    return mapping


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


def load_baseline_focal(args):
    if args.bf is not None:
        value = args.bf
    else:
        with open(args.rectification_json, "r") as handle:
            rectification = json.load(handle)
        projection = np.asarray(
            rectification["right_projection_matrix"], dtype=np.float64
        )
        if projection.shape != (3, 4):
            raise ValueError("right_projection_matrix must have shape 3x4")
        value = abs(float(projection[0, 3]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("The stereo baseline-focal product must be positive")
    return value


def load_image(path, torch):
    image = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].cuda()


def load_model(args, monster_root):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install the MonSter Python environment first") from error

    if not torch.cuda.is_available():
        raise RuntimeError("MonSter depth estimation requires CUDA")

    original_directory = Path.cwd()
    sys.path.insert(0, str(monster_root))
    sys.path.insert(0, str(monster_root / "core"))
    try:
        os.chdir(str(monster_root))
        from core.monster import Monster
        from core.utils.utils import InputPadder

        model = Monster(args)
        checkpoint = torch.load(str(args.restore_ckpt), map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Unsupported MonSter checkpoint format")
        checkpoint = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in checkpoint.items()
        }
        model.load_state_dict(checkpoint, strict=True)
        model = model.cuda().eval()
    finally:
        os.chdir(str(original_directory))
    return torch, model, InputPadder


def save_depth(output_dir, stem, depth):
    np.save(str(output_dir / (stem + ".npy")), depth.astype(np.float32))
    depth_png = np.clip(np.rint(depth), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if not cv2.imwrite(str(output_dir / (stem + ".png")), depth_png):
        raise IOError("Could not save depth map for {}".format(stem))


def estimate_depth(args):
    monster_root = Path(args.monster_root).expanduser().resolve()
    if not (monster_root / "core" / "monster.py").is_file():
        raise FileNotFoundError("Invalid MonSter checkout: {}".format(monster_root))
    args.restore_ckpt = Path(args.restore_ckpt).expanduser().resolve()
    if not args.restore_ckpt.is_file():
        raise FileNotFoundError("MonSter checkpoint does not exist: {}".format(args.restore_ckpt))

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

    stems = sorted(left_images, key=lambda stem: natural_key(left_images[stem]))
    if args.max_frames is not None:
        stems = stems[: args.max_frames]
    output_dir = prepare_output(args.output_dir, args.overwrite)
    baseline_focal = load_baseline_focal(args)
    torch, model, input_padder = load_model(args, monster_root)

    records = []
    with torch.no_grad():
        for index, stem in enumerate(stems, 1):
            left = load_image(left_images[stem], torch)
            right = load_image(right_images[stem], torch)
            if left.shape != right.shape:
                raise ValueError("Stereo image sizes differ for {}".format(stem))
            padder = input_padder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            disparity = model(left, right, iters=args.valid_iters, test_mode=True)
            disparity = padder.unpad(disparity).detach().cpu().numpy().squeeze()
            valid = np.isfinite(disparity) & (disparity > args.min_disparity)
            depth = np.zeros(disparity.shape, dtype=np.float32)
            depth[valid] = baseline_focal / disparity[valid]
            if args.max_depth is not None:
                depth[depth > args.max_depth] = 0
            save_depth(output_dir, stem, depth)
            records.append(
                {
                    "stem": stem,
                    "left": str(left_images[stem]),
                    "right": str(right_images[stem]),
                }
            )
            print("[{}/{}] {}".format(index, len(stems), stem), flush=True)

    metadata = {
        "method": "MonSter",
        "monster_repository": "https://github.com/Junda24/MonSter",
        "monster_root": str(monster_root),
        "checkpoint": str(args.restore_ckpt),
        "baseline_focal": baseline_focal,
        "depth_unit": "same linear unit as baseline_focal",
        "frame_count": len(records),
        "frames": records,
    }
    with open(str(output_dir / "depth_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Saved {} depth maps to {}".format(len(records), output_dir))


def build_parser():
    parser = argparse.ArgumentParser(description="Estimate stereo depth with MonSter")
    parser.add_argument("--monster-root", required=True)
    parser.add_argument("--restore-ckpt", required=True)
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--rectification-json")
    baseline.add_argument("--bf", type=float)
    parser.add_argument("--valid-iters", type=int, default=16)
    parser.add_argument("--min-disparity", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vitl"
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument(
        "--corr-implementation",
        choices=("reg", "alt", "reg_cuda", "alt_cuda"),
        default="reg",
    )
    parser.add_argument("--shared-backbone", action="store_true")
    parser.add_argument("--corr-levels", type=int, default=2)
    parser.add_argument("--corr-radius", type=int, default=4)
    parser.add_argument("--n-downsample", type=int, default=2)
    parser.add_argument("--slow-fast-gru", action="store_true")
    parser.add_argument("--n-gru-layers", type=int, default=3)
    parser.add_argument("--max-disp", type=int, default=192)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.valid_iters < 1:
        parser.error("--valid-iters must be positive")
    if args.min_disparity <= 0:
        parser.error("--min-disparity must be positive")
    if args.max_depth is not None and args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    try:
        estimate_depth(args)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
