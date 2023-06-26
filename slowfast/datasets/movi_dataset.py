import os
import glob
import torch

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr
from slowfast.utils.parser import load_config, parse_args

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Movi_e(Dataset):
    def __init__(self, cfg, mode='train', num_retries=1, img_glob='*.png'):
        self.root = os.path.join(cfg.MOVI.PATH, '*')
        self.img_size = 128 # IH_ cfg.SLOTS.IMG_SIZE
        self.total_dirs = sorted(glob.glob(os.path.join(cfg.MOVI.PATH, '*')))
        self.ep_len = cfg.DATA.NUM_FRAMES
        if mode == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
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

        self.transform = transforms.ToTensor()

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
        return video, torch.zeros(1), idx, torch.zeros(1)

if __name__ == "__main__":
    args = parse_args()

    args.cfg_file = '/raid/FOCUS/configs/Kinetics/SLOWFAST_4x16_R50.yaml'
    cfg = load_config(args)
    movi_dataset = Movi_e(cfg, mode='train')
    movi_dataset.__getitem__(2)
    #import pdb; pdb.set_trace()