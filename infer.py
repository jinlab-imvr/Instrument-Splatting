import json
import os
import sys
from argparse import ArgumentParser

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from arguments import ModelParams, PipelineParams
from games import gaussianModel
from renderer.gaussian_renderer import render_with_instrument_v3 as render
from scene import InstrumentScene
from utils.general_utils import safe_state
from utils.graphics_utils import build_rotation, matrix_to_quaternion
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class Pose(object):
    def __init__(self, rot, trans, alpha, theta_l, theta_r, init_optimizer=True):
        if rot.dim() == 1 and rot.numel() == 4:
            quaternion = rot
        else:
            quaternion = matrix_to_quaternion(rot)
        quaternion = F.normalize(quaternion, p=2, dim=0)

        if init_optimizer:
            self.rot = torch.nn.Parameter(quaternion.detach().clone(), requires_grad=True)
            self.trans = torch.nn.Parameter(trans.detach().clone(), requires_grad=True)
            self.alpha = torch.nn.Parameter(alpha.detach().clone(), requires_grad=True)
            self.theta_l = torch.nn.Parameter(theta_l.detach().clone(), requires_grad=True)
            self.theta_r = torch.nn.Parameter(theta_r.detach().clone(), requires_grad=True)
            self.optimizer = torch.optim.Adam(
                [
                    {"params": [self.rot], "lr": 0.001, "name": "rot"},
                    {"params": [self.trans], "lr": 0.0001, "name": "trans"},
                    {"params": [self.alpha], "lr": 0.005, "name": "alpha"},
                    {"params": [self.theta_l], "lr": 0.01, "name": "theta_l"},
                    {"params": [self.theta_r], "lr": 0.01, "name": "theta_r"},
                ],
                lr=0.0,
                eps=1e-15,
            )
        else:
            self.rot = quaternion.detach().clone()
            self.trans = trans.detach().clone()
            self.alpha = alpha.detach().clone()
            self.theta_l = theta_l.detach().clone()
            self.theta_r = theta_r.detach().clone()

    def get_pose(self):
        return (
            build_rotation(self.rot)[0],
            self.trans,
            self.alpha,
            self.theta_l,
            self.theta_r,
        )


class LPIPS(object):
    def __init__(self, device="cuda"):
        linear_weights = os.path.join(REPO_ROOT, "checkpoints", "lpips_alex_v0.1.pth")
        backbone_weights = os.path.join(
            REPO_ROOT, "checkpoints", "alexnet-owt-7be5be79.pth"
        )
        self.model = lpips.LPIPS(
            net="alex",
            pnet_rand=True,
            model_path=linear_weights,
            verbose=False,
        )
        alexnet_state = torch.load(backbone_weights, map_location="cpu")
        feature_to_slice = {0: 1, 3: 2, 6: 3, 8: 4, 10: 5}
        mapped_state = {}
        for feature_index, slice_index in feature_to_slice.items():
            for parameter_name in ("weight", "bias"):
                source_key = "features.{}.{}".format(feature_index, parameter_name)
                target_key = "slice{}.{}.{}".format(
                    slice_index, feature_index, parameter_name
                )
                mapped_state[target_key] = alexnet_state[source_key]
        self.model.net.load_state_dict(mapped_state, strict=True)
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, prediction, target):
        prediction = prediction * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self.model(prediction, target).mean()


def clone_pose_values(values):
    return tuple(value.detach().clone() for value in values)


def load_test_frames(source_path):
    split_path = os.path.join(source_path, "testing_iterations.txt")
    if not os.path.isfile(split_path):
        raise FileNotFoundError("Missing test split: {}".format(split_path))
    with open(split_path, "r") as handle:
        frame_ids = [int(line.strip()) for line in handle if line.strip()]
    if not frame_ids:
        raise ValueError("The test split is empty: {}".format(split_path))
    return frame_ids


def white_background_ground_truth(camera):
    image = camera.original_image.cuda().clone()
    semantic = camera.original_semantic_mask.cuda().permute(2, 0, 1)
    foreground = semantic.sum(dim=0, keepdim=True) > 0
    result = torch.ones_like(image)
    result[foreground.repeat(3, 1, 1)] = image[foreground.repeat(3, 1, 1)]
    return result


def render_pose(instrument, values, camera, pipe, background, pose_grad):
    rot, trans, alpha, theta_l, theta_r = values
    instrument.forward_kinematics(
        rot,
        trans,
        alpha,
        theta_l,
        theta_r,
        gs_grad=False,
        pose_grad=pose_grad,
    )
    package = render(
        camera,
        instrument,
        pipe,
        background,
        override_camera_center=trans.to(rot.dtype) * 1000.0,
        active_sh_degree=None,
        render_semantics_silhouette=False,
    )
    return package["render"].permute(2, 0, 1), package["depth"]


def refine_pose(instrument, initial_values, camera, pipe, background, target, steps):
    pose = Pose(*initial_values)
    best_values = clone_pose_values(initial_values)
    with torch.no_grad():
        initial_image, _ = render_pose(
            instrument, best_values, camera, pipe, background, pose_grad=False
        )
        best_psnr = float(psnr(initial_image[None], target[None]).mean())
    best_step = 0

    for step in range(1, steps + 1):
        pose.optimizer.zero_grad(set_to_none=True)
        image, _ = render_pose(
            instrument, pose.get_pose(), camera, pipe, background, pose_grad=True
        )
        loss = l1_loss(image, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite TTPR loss at step {}".format(step))

        candidate_psnr = float(psnr(image.detach()[None], target[None]).mean())
        if candidate_psnr > best_psnr:
            best_psnr = candidate_psnr
            best_step = step - 1
            best_values = clone_pose_values(pose.get_pose())

        loss.backward()
        pose.optimizer.step()

    return best_values, best_step


def evaluate_image(image, target, perceptual):
    with torch.no_grad():
        return {
            "SSIM": float(ssim(image[None], target[None])),
            "PSNR": float(psnr(image[None], target[None]).mean()),
            "LPIPS": float(perceptual(image[None], target[None])),
        }


def save_depth(depth, output_path):
    depth_array = depth.detach().cpu().numpy().astype(np.float32)
    np.save(output_path + ".npy", depth_array)
    finite = np.isfinite(depth_array)
    preview = np.zeros_like(depth_array, dtype=np.uint8)
    if finite.any():
        valid = depth_array[finite]
        low, high = np.percentile(valid, [1.0, 99.0])
        if high > low:
            scaled = np.clip((depth_array - low) / (high - low), 0.0, 1.0)
            preview = np.round(scaled * 255.0).astype(np.uint8)
    cv2.imwrite(output_path + ".png", preview)


def infer(dataset, pipe, gs_type, exp_name, use_ttpr, ttpr_steps, max_test_frames):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")

    experiment_path = os.path.join(dataset.exp_dir, exp_name)
    memory_pool_path = os.path.join(experiment_path, "memory_pool.pth")
    if not os.path.isfile(memory_pool_path):
        raise FileNotFoundError("Missing tracking memory pool: {}".format(memory_pool_path))

    memory_pool = {int(key): value for key, value in torch.load(memory_pool_path).items()}
    scene = InstrumentScene(
        dataset,
        gaussianModel[gs_type],
        dataset.sh_degree,
        pretrain_path=experiment_path,
        pretrain_params="trained_3dgs_param",
    )
    instrument = scene.instrument
    cameras = scene.getTrainCameras()
    test_frames = load_test_frames(scene.source_path)
    if max_test_frames is not None:
        test_frames = test_frames[:max_test_frames]

    missing_pool = sorted(set(test_frames) - set(memory_pool))
    missing_cameras = sorted(frame_id for frame_id in test_frames if frame_id >= len(cameras))
    if missing_pool:
        raise RuntimeError("Test frames absent from memory_pool.pth: {}".format(missing_pool))
    if missing_cameras:
        raise RuntimeError("Test frames absent from the camera list: {}".format(missing_cameras))

    mode = "ttpr" if use_ttpr else "no_ttpr"
    output_path = os.path.join(experiment_path, "inference_{}".format(mode))
    render_path = os.path.join(output_path, "renders_rgb")
    initial_render_path = os.path.join(output_path, "renders_rgb_before_ttpr")
    depth_path = os.path.join(output_path, "renders_depth")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    if use_ttpr:
        os.makedirs(initial_render_path, exist_ok=True)

    background = torch.ones((3,), dtype=torch.float32, device="cuda")
    perceptual = LPIPS(device="cuda")
    metrics = {}
    refined_poses = {}

    for index, frame_id in enumerate(test_frames, 1):
        camera = cameras[frame_id]
        target = white_background_ground_truth(camera)
        initial_values = clone_pose_values(memory_pool[frame_id]["pose_info"].get_pose())

        with torch.no_grad():
            initial_image, initial_depth = render_pose(
                instrument,
                initial_values,
                camera,
                pipe,
                background,
                pose_grad=False,
            )
            initial_psnr = float(psnr(initial_image[None], target[None]).mean())

        if use_ttpr:
            final_values, best_step = refine_pose(
                instrument,
                initial_values,
                camera,
                pipe,
                background,
                target,
                ttpr_steps,
            )
            save_image(
                initial_image,
                os.path.join(initial_render_path, "{:05d}.png".format(frame_id)),
            )
        else:
            final_values = initial_values
            best_step = 0

        if best_step == 0:
            image, depth = initial_image, initial_depth
        else:
            with torch.no_grad():
                image, depth = render_pose(
                    instrument,
                    final_values,
                    camera,
                    pipe,
                    background,
                    pose_grad=False,
                )
        frame_metrics = evaluate_image(image, target, perceptual)
        frame_metrics.update(
            {
                "frame_id": frame_id,
                "TTPR": use_ttpr,
                "TTPR_steps": ttpr_steps if use_ttpr else 0,
                "best_step": best_step,
                "initial_PSNR": initial_psnr,
                "PSNR_gain": frame_metrics["PSNR"] - initial_psnr,
            }
        )
        metrics[camera.image_name] = frame_metrics
        refined_poses[frame_id] = tuple(value.detach().cpu() for value in final_values)

        save_image(image, os.path.join(render_path, "{:05d}.png".format(frame_id)))
        save_depth(depth, os.path.join(depth_path, "{:05d}".format(frame_id)))
        print(
            "[{}/{}] frame={} PSNR {:.3f} -> {:.3f} ({:+.3f}), best_step={}".format(
                index,
                len(test_frames),
                frame_id,
                initial_psnr,
                frame_metrics["PSNR"],
                frame_metrics["PSNR_gain"],
                best_step,
            ),
            flush=True,
        )

    metric_names = ("SSIM", "PSNR", "LPIPS")
    summary = {
        "TTPR": use_ttpr,
        "TTPR_steps": ttpr_steps if use_ttpr else 0,
        "frame_count": len(metrics),
        "evaluated_frames": test_frames,
        "mean": {
            name: sum(record[name] for record in metrics.values()) / len(metrics)
            for name in metric_names
        },
        "mean_PSNR_gain": sum(record["PSNR_gain"] for record in metrics.values())
        / len(metrics),
    }

    with open(os.path.join(output_path, "image_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    with open(os.path.join(output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if use_ttpr:
        torch.save(refined_poses, os.path.join(output_path, "refined_pose_values.pth"))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Saved inference results to {}".format(output_path))


if __name__ == "__main__":
    parser = ArgumentParser(description="Instrument-Splatting++ inference and evaluation")
    parser.add_argument("--gs_type", type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--ttpr_steps", type=int, default=200)
    parser.add_argument("--max_test_frames", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    ttpr_group = parser.add_mutually_exclusive_group()
    ttpr_group.add_argument("--ttpr", dest="ttpr", action="store_true")
    ttpr_group.add_argument(
        "--no-ttpr", "--no_ttpr", dest="ttpr", action="store_false"
    )
    parser.set_defaults(ttpr=True)

    model_params = ModelParams(parser)
    preliminary_args, _ = parser.parse_known_args(sys.argv[1:])
    model_params.num_splats = preliminary_args.num_splats
    model_params.gs_type = preliminary_args.gs_type
    pipeline_params = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    if args.ttpr and args.ttpr_steps < 1:
        parser.error("--ttpr_steps must be positive when TTPR is enabled")
    if args.max_test_frames is not None and args.max_test_frames < 1:
        parser.error("--max_test_frames must be positive")

    safe_state(args.quiet)
    infer(
        model_params.extract(args),
        pipeline_params.extract(args),
        args.gs_type,
        args.exp_name,
        args.ttpr,
        args.ttpr_steps,
        args.max_test_frames,
    )
