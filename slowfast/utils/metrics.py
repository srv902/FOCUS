#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import numpy as np
from scipy.special import comb

def compute_ari(table):
    """
    Compute ari, given the index table
    :param table: (r, s)
    :return:
    """

    # (r,)
    a = table.sum(axis=1)
    # (s,)
    b = table.sum(axis=0)
    n = a.sum()

    comb_a = comb(a, 2).sum()
    comb_b = comb(b, 2).sum()
    comb_n = comb(n, 2)
    comb_table = comb(table, 2).sum()

    if (comb_b == comb_a == comb_n == comb_table):
        # the perfect case
        ari = 1.0
    else:
        ari = (
                (comb_table - comb_a * comb_b / comb_n) /
                (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
        )

    return ari


def compute_mask_ari(mask0, mask1):
    """
    Given two sets of masks, compute ari
    :param mask0: ground truth mask, (N0, D)
    :param mask1: predicted mask, (N1, D)
    :return:
    """

    # will first need to compute a table of shape (N0, N1)
    # (N0, 1, D)
    mask0 = mask0[:, None].byte()
    # (1, N1, D)
    mask1 = mask1[None, :].byte()
    # (N0, N1, D)
    agree = mask0 & mask1
    # (N0, N1)
    table = agree.sum(dim=-1)

    return compute_ari(table.numpy())

def evaluate_ari(true_mask, pred_mask):
    """
    :param
        true_mask: (B, N0, D)
        pred_mask: (B, N1, D)
    :return: average ari
    """
    from torch import arange as ar

    B, K, D = pred_mask.size()

    # max_index (B, D)
    max_index = torch.argmax(pred_mask, dim=1)

    # get binarized masks (B, N1, D)
    pred_mask = torch.zeros_like(pred_mask)
    pred_mask[ar(B)[:, None], max_index, ar(D)[None, :]] = 1.0

    aris = 0.
    for b in range(B):
        aris += compute_mask_ari(true_mask[b].detach().cpu(), pred_mask[b].detach().cpu())

    avg_ari = aris / B
    return avg_ari


def evaluate_mbo(true_mask, pred_mask):
    """
    Compute overlap between gt masks and the predicted masks!
    
    Steps:

    1. each gt mask is matched with predicted masks and the masks with the maximum overlap is selected!

    2. Step 1 is performed for all gt masks and the IoU is averaged.

    3. NOTE: the segmentations could be instance based or class based and in the case of instance based, the masks 
       are compared with gt instances and mBO is computed!

    """
    avg_mbo = 0.

    return avg_mbo

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]



def multitask_topks_correct(preds, labels, ks=(1,)):
    """
    Args:
        preds: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        ks: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(ks))
    task_count = len(preds)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    all_correct = all_correct.to(preds[0].device)
    for output, label in zip(preds, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    multitask_topks_correct = [
        torch.ge(all_correct[:k].float().sum(0), task_count).float().sum(0) for k in ks
    ]

    return multitask_topks_correct


def multitask_topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
   """
    num_multitask_topks_correct = multitask_topks_correct(preds, labels, ks)
    return [(x / preds[0].size(0)) * 100.0 for x in num_multitask_topks_correct]