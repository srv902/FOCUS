"""
evaluation for slot based model

TODO: metric support such as FG-ARI, mBO, MSE

"""

import numpy as np
import os
import pickle
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter
import slowfast.utils.metrics as metrics

logger = logging.get_logger(__name__)


def slot_eval(cfg):
    """
    Test slot model with video clip as input and evaluate on 
    1. action recognition
    2. video instance segmentation
    3. video object segmentation

    and include metrics such as ,

    1. FG-ARI
    2. mBO

    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)
    # exit()

    # Create video testing loaders.
    eval_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(eval_loader)))

    # from the eval_fgari_video # NOTE: used for multiple models of different seeds
    # models = []
    # for path in args.trained_model_paths:
    #     model = STEVE(args)
    #     state_dict = torch.load(path, map_location='cpu')
    #     model.load_state_dict(state_dict)
    #     model = model.cuda()
    #     models += [model]

    models = [model]
    with torch.no_grad():
        for model in models:
            model.eval()

        fgaris = []
        mbos = []
        for batch, (video, true_masks) in enumerate(eval_loader):
            video = video.cuda()

            fgaris_b = []
            mbos_b   = []
            for model in models:
                _, _, pred_masks_b_m = model.encode(video)

                # print("check the gt and pred masks! ")
                # print(true_masks.shape)     # 25 segments, size 64x64
                # print(pred_masks_b_m.shape) # 15 slots/segments, size 64x64, batch size
                # 64, 2, 25, 1, 64, 64
                # batch_size, num_iters, num_slots/num_segs, 1, h, w
                
                # check 1
                # temp1 = true_masks.permute(0, 2, 1, 3, 4, 5) # 64, 25, 2, 1, 64, 64
                # print("temp1 shape after permuting  >> ", temp1.shape)
                # temp1 = temp1[:, 1:]
                # print("temp1 shape after indexing   >> ", temp1.shape)
                # temp1 = temp1.flatten(start_dim=2)
                # print("temp1 shape after flattening >> ", temp1.shape)
                # print("temp1 >> ", temp1.shape)
                # exit()

                # FG-ARI
                # omit the BG segment i.e. the 0-th segment from the true masks as follows.
                fgari_b_m = 100 * metrics.evaluate_ari(true_masks.permute(0, 2, 1, 3, 4, 5)[:, 1:].flatten(start_dim=2),
                                            pred_masks_b_m.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
                fgaris_b += [fgari_b_m]

                # mBO
                # compute mbo # TODO: instance based (CoCo) and class based (MoVi-e)
                # NOTE: [:, 1:] removed to compare with background segments as well 
                # mbos_b_m = 100 * metrics.evaluate_mbo(true_masks.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2),
                #                             pred_masks_b_m.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
                # mbos_b  += [mbos_b_m]

            # append per batch results
            fgaris += [fgaris_b]
            # mbos   += [mbos_b]

            # print results (fg-ari)
            fgaris_numpy = np.asarray(fgaris)
            mean_ari = fgaris_numpy.mean(axis=0).mean()
            stddev_ari = fgaris_numpy.mean(axis=0).std()

            # print results (mbo)
            # mbos_numpy = np.asarray(mbos)
            # mean_mbo = mbos_numpy.mean(axis=0).mean()
            # stddev_mbo = mbos_numpy.mean(axis=0).std()

            print(f"Done batches {batch + 1}. Over {len(models)} seeds, \t FG-ARI MEAN = {mean_ari:.3f} \t STD = {stddev_ari:.3f} .")
            # print(f"Done batches {batch + 1}. Over {len(models)} seeds, \t    mBO MEAN = {mean_mbo:.3f} \t STD = {stddev_mbo:.3f} .")


if __name__ == "__main__":
    pass
