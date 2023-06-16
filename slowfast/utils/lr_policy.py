#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Learning rate policy."""

import math

def cosine_anneal(step, start_value, final_value, start_step, final_step):

    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b

    return value

def linear_warmup(step, start_value, final_value, start_step, final_step):

    assert start_value <= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b

    return value

def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    base_lrs = {'lr':cfg.SOLVER.BASE_LR}
    if cfg.SOLVER.ORVIT_BASE_LR > 0:
        base_lrs['orvit_lr'] = cfg.SOLVER.ORVIT_BASE_LR
    ret = {}
    for name, base_lr in base_lrs.items():
        lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch, base_lr=base_lr)
        # Perform warm up.
        if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
            lr_start = cfg.SOLVER.WARMUP_START_LR
            lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(
                cfg, cfg.SOLVER.WARMUP_EPOCHS
            )
            alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
            lr = cur_epoch * alpha + lr_start
        ret[name] = lr
    return ret


def lr_func_cosine(cfg, cur_epoch, base_lr = None):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    if base_lr is None:
        base_lr = cfg.SOLVER.BASE_LR
    offset = cfg.SOLVER.WARMUP_EPOCHS if cfg.SOLVER.COSINE_AFTER_WARMUP else 0.0
    assert cfg.SOLVER.COSINE_END_LR < base_lr
    return (
        cfg.SOLVER.COSINE_END_LR
        + (base_lr - cfg.SOLVER.COSINE_END_LR)
        * (
            math.cos(
                math.pi * (cur_epoch - offset) / (cfg.SOLVER.MAX_EPOCH - offset)
            )
            + 1.0
        )
        * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch, base_lr = None):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    if base_lr is None:
        base_lr = cfg.SOLVER.BASE_LR
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.LRS[ind] * base_lr


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]
