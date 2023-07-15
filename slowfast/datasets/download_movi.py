"""
movi-e download script
"""

import os
import argparse

import torch
import tensorflow_datasets as tfds
import torchvision.utils as vutils
from torchvision import transforms

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', default='/media/free2_data1/saurav/movi_e')
parser.add_argument('--level', default='e')
parser.add_argument('--split', default='train')
parser.add_argument('--version', default='1.0.0')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--dwn_opt', type=str, default='m', choices=['i', 'm', 'i+m'])
parser.add_argument('--max_num_objs', type=int, default=25)

args = parser.parse_args()

ds, ds_info = tfds.load(f"movi_{args.level}/{args.image_size}x{args.image_size}:{args.version}", data_dir="gs://kubric-public/tfds", with_info=True)
train_iter = iter(tfds.as_numpy(ds[args.split]))

to_tensor = transforms.ToTensor()

b = 0

if args.dwn_opt == 'i':
    print("Only rgb images are downloaded ..")
elif args.dwn_opt == 'm':
    print("Only segmentation masks are downloaded ..")
elif args.dwn_opt == 'i+m':
    print("Both rgb and segmentation masks are downloaded ..")

for record in train_iter:
    video_name = f"{b:08}"
    if 'i' in args.dwn_opt:
        video = record['video']
        T, *_ = video.shape
        rgb_path_vid = os.path.join(args.out_path, "data", f"{b:08}")
        os.makedirs(rgb_path_vid, exist_ok=True)
    
    if 'm' in args.dwn_opt:
        masks = record['segmentations']
        T = len(os.listdir(os.path.join(args.out_path, "data", f"{b:08}")))
        mask_path_vid = os.path.join(args.out_path, "masks", f"{b:08}")
        os.makedirs(mask_path_vid, exist_ok=True)

    print(f"Processing video {video_name} of length {T} ..")

    # # setup dirs
    # path_vid = os.path.join(args.out_path, f"{b:08}")
    # os.makedirs(path_vid, exist_ok=True)

    for t in range(T):
        if 'i' in args.dwn_opt:
            img = video[t]
            img = to_tensor(img)
            vutils.save_image(img, os.path.join(rgb_path_vid, f"{t:08}_image.png"))

        if 'm' in args.dwn_opt:
            for n in range(args.max_num_objs):
                mask = (masks[t] == n).astype(float)
                mask = torch.Tensor(mask).permute(2, 0, 1)
                vutils.save_image(mask, os.path.join(mask_path_vid, f'{t:08}_mask_{n:02}.png'))        

    b += 1