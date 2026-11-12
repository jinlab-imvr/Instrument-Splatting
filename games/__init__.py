#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

from arguments import OptimizationParams
from arguments_games import (
    OptimizationParamsMesh,
    OptimizationParamsInstrument
)

from scene.gaussian_model import GaussianModel
from games.mesh_splatting.scene.gaussian_mesh_model_3dgs import GaussianMeshModel

optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
    "gs_mesh": OptimizationParamsMesh,
    "instrument_mesh": OptimizationParamsInstrument,
}

gaussianModel = {
    "gs": GaussianModel,
    "gs_mesh": GaussianMeshModel,
    "instrument_mesh": GaussianMeshModel
}

gaussianModelRender = {
    "gs": GaussianModel,
    "gs_mesh": GaussianMeshModel,
}
