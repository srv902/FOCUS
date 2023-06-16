#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys
sys.path = [x for x  in sys.path if not (os.path.isdir(x) and 'slowfast' in os.listdir(x))]
sys.path.append(os.getcwd())

import slowfast
assert slowfast.__file__.startswith(os.getcwd()), f"sys.path: {sys.path}, slowfast.__file__: {slowfast.__file__}"

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from slot_train_net import slot_train
from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    # print("here I am fine >> ")

    cfg = assert_and_infer_cfg(cfg)
    # print(cfg)
    # print("<<<<<<<<<<<<<< After assert and infer >>> ")

    # exit()

    if cfg.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if cfg.TRAIN.ARCH == 'ar':
            launch_job(cfg=cfg, init_method=args.init_method, func=train)
        elif cfg.TRAIN.ARCH == 'slots':
            launch_job(cfg=cfg, init_method=args.init_method, func=slot_train)
        else:
            pass

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # Run demo.
    if cfg.DEMO.ENABLE:
        demo(cfg)


if __name__ == "__main__":
    main()
