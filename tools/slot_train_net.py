"""Train slot based method
STEVE, SLATE, DINOSAUR
"""

import math
import numpy as np
import pprint
import torch
from tqdm import tqdm
import torchvision.utils as vutils
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.utils.slot_misc as smisc
import slowfast.utils.lr_policy as lrp
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter, eval_extra_metrics, MetricTracker
# EPICTrainMeter, EPICValMeter
from slowfast.utils.multigrid import MultigridSchedule
import os

logger = logging.get_logger(__name__)


def slot_train_epoch(
        train_loader,
        model,
        optimizer,
        scaler,
        train_meter,
        cur_epoch,
        cfg,
        writer=None,
):
    """
    Perform training for one epoch.
    """

    # Enable train mode.
    model.train()
    # train_meter.iter_tic()
    data_size = len(train_loader)
    # convert to tqdm ..
    tqdm_loader = tqdm(train_loader, unit='batch')
    
    for cur_iter, (inputs, labels, _vid_idx, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        # print("size of the input video frames >>> ", inputs.shape)
        # exit()
        # compute misc variables needed for slot training ..
        global_step = cur_epoch * data_size + cur_iter

        tau = lrp.cosine_anneal(
            global_step,
            cfg.SLOTS_OPTIM.TAU_START,
            cfg.SLOTS_OPTIM.TAU_FINAL,
            0,
            cfg.SLOTS_OPTIM.TAU_STEPS,
        )

        lr_warmup_factor_enc = lrp.linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            cfg.SLOTS_OPTIM.WARMUP_STEPS)

        lr_warmup_factor_dec = lrp.linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            cfg.SLOTS_OPTIM.WARMUP_STEPS)

        lr_decay_factor = math.exp(global_step / cfg.SLOTS_OPTIM.HALF_LIFE * math.log(0.5))

        optim.set_slot_lr(optimizer,
                        cfg,
                        lr_decay_factor,
                        lr_warmup_factor_enc,
                        lr_warmup_factor_dec)

        # train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # preds = model(inputs, meta)
            (recon, cross_entropy, mse, attns) = model(inputs, tau, cfg.SLOTS.HARD)

            # reduce loss to mean ..
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()

            loss = mse + cross_entropy
            loss_dict = {'loss': loss}


        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()    
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)    
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )    
            
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        # logs ..
        if writer is not None:
            tqdm_loader.set_postfix(MODE='TRAIN', EPOCH=cur_epoch, STEP=global_step, LOSS=f'{loss.item():.5f}', MSE=f'{mse.item():.5f}')
            tqdm_loader.update()
            # NOTE: create dict as the writer is Tensorboard Writer not SummaryWriter
            _add = {"TRAIN/loss": loss.item()}
            _add.update({'TRAIN/cross_entropy':cross_entropy.item()})
            _add.update({'TRAIN/mse':mse.item()})
            _add.update({'TRAIN/tau':tau})
            _add.update({'TRAIN/lr_dvae':optimizer.param_groups[0]['lr']})
            _add.update({'TRAIN/lr_enc':optimizer.param_groups[1]['lr']})
            _add.update({'TRAIN/lr_dec':optimizer.param_groups[2]['lr']})

            writer.add_scalars(
                _add,
                global_step=global_step,
            )
        
        # end of interval epoch ..
        with torch.no_grad():
            if (global_step % cfg.SLOTS_OPTIM.STEP_INTERVAL == 0) and (global_step >= cfg.SLOTS_OPTIM.STEP_INTERVAL):
                # gen_video = model.module.reconstruct_autoregressive(inputs[:8])
                gen_video = (model.module if cfg.NUM_GPUS>1 else model).reconstruct_autoregressive(inputs[:8])
                frames = smisc.visualize(inputs, recon, gen_video, attns, cfg.SLOTS.NUM_SLOTS, N=8)
                # writer.add_video(f'TRAIN_recons1/steps={global_step}', frames)
                writer.add_video(frames,
                    tag=f'TRAIN_recons1/steps={global_step}',
                    global_step=global_step)

    # end of one epoch ..
    with torch.no_grad():
        # gen_video = model.module.reconstruct_autoregressive(inputs[:8])
        
        gen_video = (model.module if cfg.NUM_GPUS>1 else model).reconstruct_autoregressive(inputs[:8])
        frames = smisc.visualize(inputs, recon, gen_video, attns, cfg.SLOTS.NUM_SLOTS, N=8)
        # writer.add_video(f'TRAIN_recons2/epoch={epoch+1}', frames)
        writer.add_video(frames, 
                    tag=f'TRAIN_recons2/epoch={cur_epoch}',
                    global_step=cur_epoch)

    # NOTE: return optim specific variables. Short fix for now. 
    opd = {'tau': tau}
    
    return opd


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, opd, writer=None):
    """ Perform validation for one epoch. """

    model.eval()
    data_size = len(val_loader)
    tqdm_loader = tqdm(val_loader, unit='batch')

    for cur_iter, (inputs, labels, _vid_idx, meta) in enumerate(tqdm_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # preds = model(inputs, meta)
            (recon, cross_entropy, mse, attns) = model(inputs, opd['tau'], cfg.SLOTS.HARD)

            mse = mse.mean() # reduce loss to mean ..
            cross_entropy = cross_entropy.mean()

            loss = mse + cross_entropy
            loss_dict = {'loss': loss}

            # use the meter to get averaged loss later ..
            val_meter.update(loss)

        if writer is not None: # logs ..
            tqdm_loader.set_postfix(MODE='VAL', EPOCH=cur_epoch, LOSS=f'{loss.item():.5f}', MSE=f'{mse.item():.5f}')
            # NOTE: create dict as the writer is Tensorboard Writer not SummaryWriter
            _add = {"VAL/loss": loss.item()}
            _add.update({'VAL/cross_entropy':cross_entropy.item()})
            _add.update({'VAL/mse':mse.item()})

            writer.add_scalars(_add, global_step=cur_epoch)
        
    # end of interval epoch ..
    with torch.no_grad():
        if 50 <= cur_epoch:
            gen_video = (model.module if cfg.NUM_GPUS>1 else model).reconstruct_autoregressive(inputs[:8])
            frames = smisc.visualize(inputs, recon, gen_video, attns, cfg.SLOTS.NUM_SLOTS, N=8)
            # writer.add_video(f'TRAIN_recons1/steps={global_step}', frames)
            writer.add_video(frames,
                tag=f'VAL_recons/steps={cur_epoch}',
                global_step=global_step)

    return val_meter.avg

def slot_train(cfg):
    """
    Train a slot model with video clip as input and evaluate on 
    1. action recognition
    2. video instance segmentation
    3. video object segmentation
    """

    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)    

    # Init multigrid.
    multigrid = None

    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))    

    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    # optimizer = optim.construct_optimizer(model, cfg)
    optimizer = optim.construct_optimizer_slot(model, cfg) # NOTE: only for the slots
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    # val_meter = ValMeter(len(val_loader), cfg)
    val_meter = MetricTracker() # NOTE: to track loss and needed for checkpointing

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    # for checkpointing ..
    best_val_loss = math.inf

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        if not cfg.TRAIN.VAL_ONLY:
            opd = slot_train_epoch(
                train_loader,
                model,
                optimizer,
                scaler,
                train_meter,
                cur_epoch,
                cfg,
                writer,
            )

        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )
    

        # print("eval epoch vals >>> ", is_eval_epoch)
        # Evaluate the model on validation set.
        # if is_eval_epoch: # NOTE: assuming after training val is performed. Not tuned to specific intervals for validation step.
        val_loss = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, opd, writer)
        if cfg.TRAIN.VAL_ONLY:
            break

        # check if the model weights should be saved based on the val loss
        is_checkp_epoch = False
        print(f"val loss {val_loss} and best_val_loss {best_val_loss} >>>>>>>>>>>")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_checkp_epoch = True

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

    if writer is not None:
        writer.close()