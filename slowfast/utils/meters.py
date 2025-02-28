#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
from pickle import load
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score
from torch._C import _set_backcompat_keepdim_warn

import slowfast.datasets.ava_helper as ava_helper
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_exclusions,
    read_labelmap,
)

import slowfast.models.losses as losses

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(
            cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
        )
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

        _, self.video_idx_to_name = ava_helper.load_image_lists(
            cfg, mode == "train"
        )
        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}/{}".format(cur_iter + 1, self.overall_iters),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "loss": self.loss.get_win_median(),
                "lr": self.lr,
            }
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True):
        """
        Calculate and log the final AVA metrics.
        """
        all_preds = torch.cat(self.all_preds, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)

        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        self.full_map = evaluate_ava(
            all_preds,
            all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,
        )
        if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "map": self.full_map,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            logging.log_json_stats(stats)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            self.stats["map"] = map
        else:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.multi_loss = MultiLossMeter(cfg.LOG_PERIOD, prefix="multi_loss")
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.multi_loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        dloss, loss = prepare_loss_dict(loss)
        self.multi_loss.add_value(dloss)
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }

        stats.update(self.multi_loss.get_win_median())

        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        
        stats.update(self.multi_loss.get_global_avg())

        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        self.extra_metrices = MultiLossMeter(cfg.LOG_PERIOD, prefix='extra_meter')
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.extra_metrices.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size, extra_metrices={}):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size
        self.extra_metrices.add_value(extra_metrices)

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        stats.update(self.extra_metrices.get_win_median())
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err
        stats.update(self.extra_metrices.get_global_avg())

        logging.log_json_stats(stats)

class EPICTrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.loss_verb = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_verb_total = 0.0
        self.loss_noun = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_noun_total = 0.0
        self.lr = None
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.loss_verb.reset()
        self.loss_verb_total = 0.0
        self.loss_noun.reset()
        self.loss_noun_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.loss_verb.add_value(loss[0])
        self.loss_noun.add_value(loss[1])
        self.loss.add_value(loss[2])
        self.lr = lr
        # Aggregate stats
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.loss_verb_total += loss[0] * mb_size
        self.loss_noun_total += loss[1] * mb_size
        self.loss_total += loss[2] * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": self.mb_verb_top1_acc.get_win_median(),
            "verb_top5_acc": self.mb_verb_top5_acc.get_win_median(),
            "noun_top1_acc": self.mb_noun_top1_acc.get_win_median(),
            "noun_top5_acc": self.mb_noun_top5_acc.get_win_median(),
            "top1_acc": self.mb_top1_acc.get_win_median(),
            "top5_acc": self.mb_top5_acc.get_win_median(),
            "verb_loss": self.loss_verb.get_win_median(),
            "noun_loss": self.loss_noun.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        avg_loss_verb = self.loss_verb_total / self.num_samples
        avg_loss_noun = self.loss_noun_total / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": verb_top1_acc,
            "verb_top5_acc": verb_top5_acc,
            "noun_top1_acc": noun_top1_acc,
            "noun_top5_acc": noun_top5_acc,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "verb_loss": avg_loss_verb,
            "noun_loss": avg_loss_noun,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)


class EPICValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accuracies (over the full val set).
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        self.max_verb_top1_acc = 0.0
        self.max_verb_top5_acc = 0.0
        self.max_noun_top1_acc = 0.0
        self.max_noun_top5_acc = 0.0
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            mb_size (int): mini batch size.
        """
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.num_samples += mb_size
        self.all_preds = []
        self.all_labels = []

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": self.mb_verb_top1_acc.get_win_median(),
            "verb_top5_acc": self.mb_verb_top5_acc.get_win_median(),
            "noun_top1_acc": self.mb_noun_top1_acc.get_win_median(),
            "noun_top5_acc": self.mb_noun_top5_acc.get_win_median(),
            "top1_acc": self.mb_top1_acc.get_win_median(),
            "top5_acc": self.mb_top5_acc.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        self.max_verb_top1_acc = max(self.max_verb_top1_acc, verb_top1_acc)
        self.max_verb_top5_acc = max(self.max_verb_top5_acc, verb_top5_acc)
        self.max_noun_top1_acc = max(self.max_noun_top1_acc, noun_top1_acc)
        self.max_noun_top5_acc = max(self.max_noun_top5_acc, noun_top5_acc)
        is_best_epoch = top1_acc > self.max_top1_acc
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "verb_top1_acc": verb_top1_acc,
            "verb_top5_acc": verb_top5_acc,
            "noun_top1_acc": noun_top1_acc,
            "noun_top5_acc": noun_top5_acc,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "max_verb_top1_acc": self.max_verb_top1_acc,
            "max_verb_top5_acc": self.max_verb_top5_acc,
            "max_noun_top1_acc": self.max_noun_top1_acc,
            "max_noun_top5_acc": self.max_noun_top5_acc,
            "max_top1_acc": self.max_top1_acc,
            "max_top5_acc": self.max_top5_acc,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

        return is_best_epoch
    
    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds[0])
        self.all_labels.append(labels['verb'])


class EPICTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(self, num_videos, num_clips, num_cls, overall_iters):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        # Initialize tensors.
        self.verb_video_preds = torch.zeros((num_videos, num_cls[0]))
        self.noun_video_preds = torch.zeros((num_videos, num_cls[1]))
        self.verb_video_labels = torch.zeros((num_videos)).long()
        self.noun_video_labels = torch.zeros((num_videos)).long()
        self.metadata = np.zeros(num_videos, dtype=object)
        self.clip_count = torch.zeros((num_videos)).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.verb_video_preds.zero_()
        self.verb_video_labels.zero_()
        self.noun_video_preds.zero_()
        self.noun_video_labels.zero_()
        self.metadata.fill(0)

    def update_stats(self, preds, labels, metadata, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds[0].shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            self.verb_video_labels[vid_id] = labels[0][ind]
            self.verb_video_preds[vid_id] += preds[0][ind]
            self.noun_video_labels[vid_id] = labels[1][ind]
            self.noun_video_preds[vid_id] += preds[1][ind]
            self.metadata[vid_id] = metadata['narration_id'][ind]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    self.clip_count, self.num_clips
                )
            )
            logger.warning(self.clip_count)

        verb_topks = metrics.topk_accuracies(self.verb_video_preds, self.verb_video_labels, ks)
        noun_topks = metrics.topk_accuracies(self.noun_video_preds, self.noun_video_labels, ks)
        
        # Compute the action accuracies.	
        action_topks = metrics.multitask_topk_accuracies(
            (self.verb_video_preds, self.noun_video_preds),	
            (self.verb_video_labels, self.noun_video_labels), 
            ks
        )

        assert len({len(ks), len(verb_topks)}) == 1
        assert len({len(ks), len(noun_topks)}) == 1
        stats = {"split": "test_final"}
        for k, verb_topk in zip(ks, verb_topks):
            stats["verb_top{}_acc".format(k)] = "{:.{prec}f}".format(verb_topk, prec=2)
        for k, noun_topk in zip(ks, noun_topks):
            stats["noun_top{}_acc".format(k)] = "{:.{prec}f}".format(noun_topk, prec=2)
        for k, action_topk in zip(ks, action_topks):	
            stats["action_top{}_acc".format(k)] = "{:.{prec}f}".format(action_topk, prec=2)
        logging.log_json_stats(stats)
        return (self.verb_video_preds.numpy().copy(), self.noun_video_preds.numpy().copy()), \
               (self.verb_video_labels.numpy().copy(), self.noun_video_labels.numpy().copy()), \
               self.metadata.copy()

def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)


class MultiLossMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size, prefix = None):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.window_size = window_size
        self.prefix = prefix
        self.losses = dict()

    def add_prefix(self, d):
        if self.prefix is None: return d
        return {f"{self.prefix}_{k}":v for k,v in d.items()}

    def reset(self):
        """
        Reset the deque.
        """
        for l in self.losses.values(): l.reset()

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        value = loss_dict_to_float(value)
        for k, v in value.items():
            if k not in self.losses:
                self.losses[k] = ScalarMeter(self.window_size)
            self.losses[k].add_value(v)

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return self.add_prefix({k:v.get_win_median() for k,v in self.losses.items()})

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return self.add_prefix({k:v.get_win_avg() for k,v in self.losses.items()})

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.add_prefix({k:v.get_global_avg() for k,v in self.losses.items()})

def to_float(v):
    if isinstance(v, torch.Tensor): v = v.item()
    return v

def loss_dict_to_float(d):
    ret = {}
    for k,v in d.items():
        ret[k] = to_float(v)
    return ret

def prepare_loss_dict(loss):
    if isinstance(loss ,dict):
        dloss = loss
        if 'loss' in dloss: loss = dloss['loss']
        else: loss = sum(dloss.values())
    else:
        dloss = {}
    dloss = loss_dict_to_float(dloss)
    return dloss, to_float(loss)


##


def eval_extra_metrics(cfg, preds, extra_preds, labels, metadata):
    loss_fun = losses.get_loss_func(cfg, state = 'val')(
        reduction="mean"
    )
    loss = loss_fun(preds, labels)
    loss_dict = {'loss':loss}

    return loss_dict

# generic metric tracker ..
class MetricTracker(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.curr  = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0
        self.best_score = 0

    def update(self, val, n=1.):
        self.curr   = val
        self.sum   += val
        self.count += n
        self.avg    = self.sum / self.count

    def update_test(self, val):
        self.best_score = val