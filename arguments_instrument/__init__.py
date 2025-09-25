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

from arguments import ParamGroup


class OptimizationParamsInstrument(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.vertices_lr = 0.0016 * 1e-3 #0.0016 * 1e-3
        self.alpha_lr = 0.001 
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005 # 0.0005 
        self.rotation_lr = 0.001
        self.random_background = False
        self.use_mesh = True
        self.lambda_dssim = 0.2
        self.percent_dense = 0.01
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")
