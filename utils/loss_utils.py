#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision.utils import save_image

def huber_loss(network_output, gt, delta=1.0):
    abs_error = torch.abs(network_output - gt)
    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_normalization(network_output, gt):
    # normalize the network output and ground truth
    network_output = network_output / network_output.max()
    gt = gt / gt.max()
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def dice(input, target):
    smooth = 1e-8
    input = input.view(-1)
    target = target.view(-1)
    intersection = (input * target).sum()
    return ((2. * intersection + smooth) / (input.sum() + target.sum() + smooth))

def instrument_dice(input, target):
    wrist_pred = input[1] == 1
    wrist_gt = target[1] == 1
    wrist_dice = dice(wrist_pred, wrist_gt)
    shaft_pred = input[0] == 1
    shaft_gt = target[0] == 1
    shaft_dice = dice(shaft_pred, shaft_gt)
    gripper_pred = input[2] == 1
    gripper_gt = target[2] == 1
    gripper_dice = dice(gripper_pred, gripper_gt)
    return wrist_dice, shaft_dice, gripper_dice