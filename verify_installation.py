from __future__ import print_function

import argparse
import os
import sys
from pathlib import Path

import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_MODULES = (
    REPO_ROOT,
    REPO_ROOT / "submodules" / "depth-diff-gaussian-rasterization",
    REPO_ROOT / "submodules" / "simple-knn",
)
for module_path in LOCAL_MODULES:
    path_string = str(module_path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def verify_cuda_extensions():
    import diff_gaussian_rasterization
    from diff_gaussian_rasterization import _C as rasterizer_cuda
    from simple_knn._C import distCUDA2

    points = torch.rand((8, 3), dtype=torch.float32, device="cuda")
    distances = distCUDA2(points)
    if distances.shape != (8,) or not torch.isfinite(distances).all():
        raise RuntimeError("simple-knn CUDA smoke test failed")
    print("[OK] CUDA extensions: {}".format(diff_gaussian_rasterization.__file__))
    print("[OK] Rasterizer binary: {}".format(rasterizer_cuda.__file__))


def verify_cad_models():
    mesh_names = ("shaft", "wrist", "gripper_left", "gripper_right")
    for name in mesh_names:
        mesh_path = REPO_ROOT / "instrument_mesh" / ("transformed_{}.obj".format(name))
        mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        if not len(mesh.vertices) or not len(mesh.faces):
            raise RuntimeError("Empty CAD model: {}".format(mesh_path))
        print("[OK] CAD {}: {} vertices, {} faces".format(name, len(mesh.vertices), len(mesh.faces)))


def verify_checkpoint_files():
    required_keys = {
        "opacity",
        "alpha",
        "xyz",
        "rotation",
        "scale",
        "scaling",
        "features_dc",
        "features_rest",
    }
    model_paths = sorted((REPO_ROOT / "pretrained_models").glob("*.pth"))
    if len(model_paths) != 8:
        raise RuntimeError("Expected 8 initialized GS models, found {}".format(len(model_paths)))
    for model_path in model_paths:
        state = torch.load(str(model_path), map_location="cpu")
        missing_keys = required_keys.difference(state)
        if missing_keys:
            raise RuntimeError("{} is missing {}".format(model_path, sorted(missing_keys)))
        print("[OK] GS checkpoint {}: {} Gaussians".format(model_path.name, state["xyz"].shape[0]))

    auxiliary_paths = (
        REPO_ROOT / "checkpoints" / "checkpoint_ToolTipNet.pth",
        REPO_ROOT / "checkpoints" / "alexnet-owt-7be5be79.pth",
        REPO_ROOT / "checkpoints" / "lpips_alex_v0.1.pth",
        REPO_ROOT / "submodules" / "ml_aspanformer" / "weights" / "outdoor.ckpt",
    )
    for checkpoint_path in auxiliary_paths:
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise RuntimeError("Missing checkpoint: {}".format(checkpoint_path))
        print("[OK] Bundled checkpoint: {}".format(checkpoint_path.relative_to(REPO_ROOT)))


def verify_aspanformer():
    from submodules.ml_aspanformer.src.ASpanFormer.aspanformer import ASpanFormer
    from submodules.ml_aspanformer.src.config.default import get_cfg_defaults
    from submodules.ml_aspanformer.src.utils.misc import lower_config

    config_path = REPO_ROOT / "submodules" / "ml_aspanformer" / "configs" / "aspan" / "outdoor" / "aspan_test.py"
    weights_path = REPO_ROOT / "submodules" / "ml_aspanformer" / "weights" / "outdoor.ckpt"
    config = get_cfg_defaults()
    config.merge_from_file(str(config_path))
    matcher = ASpanFormer(config=lower_config(config)["aspan"])
    state = torch.load(str(weights_path), map_location="cpu")["state_dict"]
    matcher.load_state_dict(state, strict=False)
    print("[OK] ASpanFormer checkpoint: {} tensors".format(len(state)))


def verify_tipnet():
    from games.mesh_splatting.scene.network import KeyPointNet

    checkpoint_path = REPO_ROOT / "checkpoints" / "checkpoint_ToolTipNet.pth"
    state = torch.load(str(checkpoint_path), map_location="cpu")["model_state_dict"]
    model = KeyPointNet(n_kp=2, height=512, width=640, use_gpu=True)
    model.load_state_dict(state)
    model = model.cuda().eval()
    with torch.no_grad():
        image = torch.zeros((1, 3, 512, 640), dtype=torch.float32, device="cuda")
        mask = torch.ones((1, 1, 128, 160), dtype=torch.float32, device="cuda")
        keypoints, _ = model(image, mask)
    if keypoints.shape != (1, 2, 2) or not torch.isfinite(keypoints).all():
        raise RuntimeError("Gripper-Tip Net forward smoke test failed")
    print("[OK] Gripper-Tip Net forward: {}".format(tuple(keypoints.shape)))
    del model, state, image, mask, keypoints
    torch.cuda.empty_cache()


def verify_lpips():
    from infer import LPIPS

    metric = LPIPS(device="cuda")
    with torch.no_grad():
        image = torch.zeros((1, 3, 64, 64), dtype=torch.float32, device="cuda")
        score = metric(image, image)
    if not torch.isfinite(score):
        raise RuntimeError("LPIPS forward smoke test failed")
    print("[OK] Bundled LPIPS forward: {:.6f}".format(float(score)))
    del metric, image, score
    torch.cuda.empty_cache()


def verify_instrument_models():
    from games import gaussianModel
    from scene import InstrumentGaussianWrapper

    variants = ("pretrain_param_3dgs", "pretrain_3dgs_dense_param")
    for variant in variants:
        wrapper = InstrumentGaussianWrapper(
            gaussianModel["gs_mesh"],
            3,
            variant,
            pretrain_path=str(REPO_ROOT / "pretrained_models"),
        )
        counts = {
            name: int(part.gaussian_model.get_xyz.shape[0])
            for name, part in wrapper.instrument.part_dict.items()
        }
        if not all(counts.values()):
            raise RuntimeError("Empty instrument model for {}".format(variant))
        print("[OK] Instrument model {}: {}".format(variant, counts))
        del wrapper
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Validate bundled Instrument-Splatting++ dependencies")
    parser.add_argument("--cpu-only", action="store_true", help="Skip CUDA model and extension smoke tests")
    args = parser.parse_args()

    verify_cad_models()
    verify_checkpoint_files()
    verify_aspanformer()
    if not args.cpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --cpu-only for asset-only validation")
        verify_cuda_extensions()
        verify_tipnet()
        verify_lpips()
        verify_instrument_models()
    print("All bundled dependencies passed validation.")


if __name__ == "__main__":
    main()
