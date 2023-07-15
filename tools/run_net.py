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
# from slot_train_net import slot_train
from steve_train_net import slot_train
from steve_eval_net import slot_eval
from visualization import visualize

# NOTE: fix to avoid redundant [DEBUG] PngImagePlugin.py: 201 errors
import logging as lg
lg.getLogger('PIL').setLevel(lg.WARNING)

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # set path to save the checkpoints
    cfg.EXP.NAME = args.exp_name
    cfg.EXP.PATH = os.path.join(cfg.OUTPUT_DIR, args.exp_name)

    if cfg.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if cfg.TRAIN.METHOD == 'sup':
            launch_job(cfg=cfg, init_method=args.init_method, func=train)
        elif cfg.TRAIN.METHOD == 'slots':
            launch_job(cfg=cfg, init_method=args.init_method, func=slot_train)
        else:
            pass

    # Perform testing
    if cfg.TEST.ENABLE:
        if cfg.TEST.EVAL_TASK == 'segmentation':
            # NOTE: add the segmentation evaluation code from the learned slot features ..
            launch_job(cfg=cfg, init_method=args.init_method, func=slot_eval)
        elif cfg.TEST.EVAL_TASK == 'ar':
            # NOTE: default setting when no slot based learning is used. the standard way of testing action recognition tasks!
            launch_job(cfg=cfg, init_method=args.init_method, func=test)
        else:
            pass

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
