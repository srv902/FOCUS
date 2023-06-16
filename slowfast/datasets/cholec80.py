"""
Dataset and Dataloader to work on the Cholec80 phase and tool instance tasks 
"""

import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

def read_pkl(filename='test'):
    # print("Processing file >> ", filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

@DATASET_REGISTRY.register()
class Cholec80(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train', num_retries=1):
        assert mode in [
            'train',
            'val',
            'test'
        ], f"Split {mode} not supported for Cholec80"
        self.mode = mode
        self.cfg  = cfg
        self._num_retries = num_retries

        if mode == 'train':
            self.pkl_file = os.path.join(cfg.CHOLEC.PATH, mode, cfg.CHOLEC.TRAIN_PKL)
        elif mode == 'val':
            self.pkl_file = os.path.join(cfg.CHOLEC.PATH, mode, cfg.CHOLEC.VAL_PKL)
        elif mode == 'test':
            self.pkl_file = os.path.join(cfg.CHOLEC.PATH, mode, cfg.CHOLEC.TEST_PKL)

        logger.info("Constructing Cholec80 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader
        """
        self.data = read_pkl(self.pkl_file)
        video_list = list(self.data.keys())[-1:]
        # print("video list >>> ")
        # print(video_list)
        # NOTE: to debug reduce the video list to only one video

        self.map  = {}
        # create a big list of frame sequence of all videos ..
        
        self._frame_list = []
        # print("list of video in mode ", self.mode)
        # print(self.data.keys())
        for v in video_list:
            # print("video in use >>> ", v)
            frame_list = self.data[v][:50]


            # create a mapping from the list index to the frame ids ..
            self.map[v] = {}
            for i,k in enumerate(frame_list):
                # print("one sample at a time >> ")
                # print(k)
                self.map[v][k['Frame_id']] = i 
                # exit()

            # print("check the map var ?? ")
            # print(self.map)
            # exit()

            # print(f"number of frames in video {v} > {len(frame_list)}")
            # print("Frame list for one random video >> ")
            # print(frame_list)
            # exit()
            self._frame_list.extend(frame_list)
            # print("len of frame_list >> ", len(self._frame_list), v)

        self.num_frames = len(self._frame_list)
        # print("num frames in the global list >> ", self.num_frames)

        # print("one sample >> ")
        # print(self._frame_list[0])
        # exit()

        logger.info(f"Cholec80 dataloader constructed (size: {self.num_frames}) from {self.pkl_file}")
        # self.transforms = self.get_transforms()
        # self._gen_data()

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        temporal_sample_index = (
            -1
            if self.mode in ["train", "val"]
            else self._spatial_temporal_idx[index]
            // self.cfg.TEST.NUM_SPATIAL_CROPS
        )

        num_frames = self.cfg.DATA.NUM_FRAMES
        # sampling_rate = utils.get_random_sampling_rate(
        #     self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
        #     self.cfg.DATA.SAMPLING_RATE,
        # )
        sampling_rate = self.cfg.DATA.SAMPLING_RATE

        # NOTE: the actual sequence of frames based on the sampling rate is already taken care of in pkl
        # For each index, get the video id and fetch the sequence from the self.data and then build a clip 

        index_record = self._frame_list[index]
        # video_id_select = f"video{str(index_record['unique_id'])[:2]}"
        # f"{int(t2):02}"
        video_id_select = f"video{int(str(index_record['unique_id'])[:-8]):02}"
        # print("video_id_select >> ", video_id_select)

        # if video_id_select == 'video70':
        #     print("index record")
        #     print(index_record)
        #     print("index >>> ", index)

        # exit()

        frame_id = index_record['Frame_id']

        # NOTE: start iterating from the self.data and build the clip .. pad if necessary ..
        seq = []
        # print("video id select in use >>>>>>>>>>>>>> ", video_id_select)
        end_index = self.map[video_id_select][frame_id]
        for k in range(end_index, -1, -1):
            rec = self.data[video_id_select][k]
            seq.append(rec)

            if len(seq) == num_frames:
                break

        # if the clip len is not equal to the num frames >>
        if len(seq) < num_frames:
            gap = num_frames - len(seq)
            to_repeat = seq[0]
            seq = [to_repeat] * gap + seq


        # add frame_path key which denotes the path to the frame ids ..
        for i,j in enumerate(seq):
            seq[i]['frame_path'] = os.path.join(self.cfg.DATA.PATH_PREFIX, video_id_select, f"{j['Frame_id']}.jpg")

        # print("Seq >> ")
        # print(seq)

        assert len(seq) == num_frames, "seq does not contain num_frames frames!!!"

        return seq
    
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        seq = self.get_seq_frames(index)
        frames = torch.as_tensor(
            utils.retry_load_images(
                [ frame['frame_path'] for frame in seq],
                self._num_retries,
            )
        )

        # NOTE: to design the label based on the type of the downstream task >> phase or tool presence ..
        # label = utils.aggregate_labels(
        #     [self._labels[index][i] for i in range(seq[0], seq[-1] + 1)]
        # )
        # label = torch.as_tensor(
        #     utils.as_binary_vector(label, self.cfg.MODEL.NUM_CLASSES)
        # )
        
        # Assuming phase labels for now .. NOTE: to add tool presence labels later ..
        label = [frame['Phase_gt'] for frame in seq]

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        # frames = frames.permute(3, 0, 1, 2)

        # T H W C -> T C H W
        frames = frames.permute(0, 3, 1, 2)

        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )

        # print("size of the frames >>> ", frames.shape)
        # frames = utils.pack_pathway_output(self.cfg, frames)
        # print("size of the frames before return >>> ", frames.shape)

        # reverse input channel ..
        # frames = frames[[2, 1, 0], :, :, :]

        return frames, label, index, {}

    def __len__(self):
        return len(self._frame_list)

if __name__ == "__main__":
    from types import SimpleNamespace 
    args = {}
    args['mode'] = 'train'
    args['seg_path']  = '/home/ssharma/projects/repos/sam-ssg/video_masks_cholec80'
    args['data_path'] = '/home/ssharma/datasets/surgical/cholec/cholec80'
    args['pkl_path']  = '/home/ssharma/projects/repos/sam-ssg/datasets/cholec80/labels' 
    args['train_pkl_file'] = '1fps_100_0.pickle'
    args['val_pkl_file']   = '1fps.pickle'
    args['test_pkl_file']  = '1fps.pickle'
    args['feat_h'] = 8
    args['feat_w'] = 14
    args['img_h'] = 224 #224
    args['img_w'] = 224 #224
    args['h'] = 256
    args['w'] = 448
    args['vit_type'] = 'b' # b & h
    args['vlist'] = 37
    args['opath'] = 'video_masks_cholec80'
    args['ipath'] = 'seg_json_cholec80'
    args['server'] = 'jz'
    args['ds_task'] = 'phase-recognition'

    args = SimpleNamespace(**args)

    ds = Cholec80(args)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    for i, (img, lbl, sm, sm_nhw) in enumerate(dl):
        print("img shape >> ", img.shape)
        print("seg mask shape >> ", sm.shape)
        print("seg mask shape >> ", sm_nhw.shape)
        sm = sm.squeeze(0) #.permute(1, 0)
        sm = sm.numpy()
        plt.imshow(sm)
        plt.show()
        exit()