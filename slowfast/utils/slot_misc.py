"""
utils needed for the slot mechanism to work and display slot results on the tensorboard
"""

import os
import cv2
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

def visualize(video, recon_dvae, recon_tf, attns, num_slots=4, N=8,):
    B, T, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :]

        # tile
        tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(num_slots + 3), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames
