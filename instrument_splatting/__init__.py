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

from arguments import OptimizationParamsInstrument



from instrument_splatting.scene.gaussian_model import GaussianModel
from instrument_splatting.scene.gaussian_instrument_model import GaussianInstrumentModel

optimizationParamTypeCallbacks = {
    "gs_instrument": OptimizationParamsInstrument,
    # "instrument_mesh": OptimizationParamsInstrument,
}

gaussianModel = {
    "gs_instrument": GaussianInstrumentModel,
    # "instrument_mesh": GaussianInstrumentModel
}

gaussianModelRender = {
    "gs_instrument": GaussianInstrumentModel,
    # "instrument_mesh": GaussianInstrumentModel,
}
