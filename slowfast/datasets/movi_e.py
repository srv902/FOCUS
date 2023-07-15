"""
PyTorch dataset class for MoVi-E 
"""

import os
import glob
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

import torch
from torchvision import transforms as T

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

@DATASET_REGISTRY.register()
class Movi_e(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.root       = cfg.DATA.PATH_TO_DATA_DIR
        self.img_size   = cfg.DATA.TRAIN_CROP_SIZE
        self.total_dirs = sorted(glob.glob(self.root))

        self.ep_len     = cfg.SLOTS.NUM_ITERS
        img_glob        = cfg.DATA.GLOB_EXP

        if mode == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
            # self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.01)]
        elif mode == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif mode == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        else:
            pass

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video

# Video and Masks together for MoVi-E
@DATASET_REGISTRY.register()
class Movi_e_with_masks(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.root       = cfg.DATA.PATH_TO_DATA_DIR
        self.img_size   = cfg.DATA.TRAIN_CROP_SIZE
        self.total_dirs = sorted(glob.glob(self.root))

        self.ep_len = cfg.SLOTS.NUM_ITERS
        num_segs    = cfg.DATA.NUM_SEGS
        img_glob    = cfg.DATA.GLOB_EXP

        # chunk into episodes
        self.episodes_rgb  = []
        self.episodes_mask = []
        for dir in self.total_dirs:
            frame_buffer = []
            mask_buffer  = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for image_path in image_paths:
                p = Path(image_path)
                frame_buffer.append(p)
                
                # add masks path
                parent = str(p.parent).replace('frames', 'masks')
                # print("parent >> ", parent)
                # for n in range(num_segs):
                #     print(os.path.join(parent, f"{p.stem.split('_')[0]}_mask_{n:02}.png"))
                # exit()

                mask_buffer.append([
                    os.path.join(parent, f"{p.stem.split('_')[0]}_mask_{n:02}.png") for n in range(num_segs)
                ])

                if len(frame_buffer) == self.ep_len:
                    self.episodes_rgb.append(frame_buffer)
                    self.episodes_mask.append(mask_buffer)
                    frame_buffer = []
                    mask_buffer  = []

        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.episodes_rgb)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes_rgb[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)

        masks = []
        for mask_locs in self.episodes_mask[idx]:
            frame_masks = []
            for mask_loc in mask_locs:
                image = Image.open(mask_loc).convert('1')
                image = image.resize((self.img_size, self.img_size))
                tensor_image = self.transform(image)
                frame_masks += [tensor_image]
            frame_masks = torch.stack(frame_masks, dim=0)
            masks += [frame_masks]
        masks = torch.stack(masks, dim=0)

        return video, masks

if __name__ == "__main__":
    pass
