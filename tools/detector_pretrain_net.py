# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import datetime
import os
import time

import gpustat
import torch

from pysgg.config import cfg
from pysgg.data import make_data_loader
from pysgg.engine.inference import inference
from pysgg.engine.trainer import reduce_loss_dict
from pysgg.modeling.detector import build_detection_model
from pysgg.solver import make_lr_scheduler
from pysgg.solver import make_optimizer
from pysgg.utils.checkpoint import DetectronCheckpointer
from pysgg.utils.collect_env import collect_env_info
from pysgg.utils.comm import synchronize, get_rank, all_gather
from pysgg.utils.logger import setup_logger, TFBoardHandler_LEVEL
from pysgg.utils.metric_logger import MetricLogger
from pysgg.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
from tools.relation_train_net import show_params_status

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, logger):
    info = gpustat.core.GPUStatCollection.new_query()
    placeholder = None
    mem_size = info.jsonify()['gpus'][get_rank()]
    if mem_size['memory.used'] / mem_size['memory.total'] < 0.5:
        if mem_size['memory.total'] < 7000:
            placeholder = torch.zeros((90000, 8196), device=torch.torch.device("cuda"))
        else:
            placeholder = torch.zeros((180000, 8196), device=torch.torch.device("cuda"))

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    logger.info(str(model))

    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))
    scheduler = make_lr_scheduler(cfg, optimizer)

    output_dir = cfg.OUTPUT_DIR

    load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

    arguments = {}
    arguments["iteration"] = 0

    # to load the model from normal float32 type, you need to load the param before
    # transform model to mixed-precision
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    if cfg.MODEL.PRETRAINED_DETECTOR_CKPT != "":
        checkpointer.load_weight_partially(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
    else:
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,
                                                  with_optim=False,
                                                  update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD,
                                                  load_mapping=load_mapping)
        arguments.update(extra_checkpoint_data)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    logger.info("trainable models:")
    logger.info(show_params_status(model))


    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed)

    if placeholder is not None:
        placeholder = placeholder.to("cpu")
        del placeholder

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        model.train()

        if any(len(target) < 1 for target in targets):
            logger.error(
                f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))

        if iteration % 30 == 0:
            logger.log(TFBoardHandler_LEVEL, (meters.meters, iteration))

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "\nelapsed time: {elapsed_time}\n",
                        "eta: {eta}\n",
                        "iter: {iter}/{max_iter}\n",
                        "{meters}",
                        "lr: {lr:.6f}\n",
                        "max mem: {memory:.0f}\n",
                    ]
                ).format(
                    eta=eta_string,
                    elapsed_time=elapsed_time,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    max_iter=max_iter,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        val_result_value = None  # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed)
            if get_rank() == 0:
                for each_ds_eval in val_result[0]:
                    for each_evalator_res in each_ds_eval[1]:
                        logger.log(TFBoardHandler_LEVEL, (each_evalator_res, iteration))

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    return model


def run_val(cfg, model, val_data_loaders, distributed):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
        )
        synchronize()
        val_result.append(dataset_result)
    print("accumulated validation results form multi-GPUs")
    val_values = []
    for each in val_result:
        if isinstance(each, tuple):
            val_values.append(each[0])
    # support for multi gpu distributed testing
    # send evaluation results to each process
    gathered_result = all_gather(torch.tensor(val_values).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result_val = float(valid_result.mean())

    del gathered_result

    return val_result, val_result_val


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(
        cfg,
        mode='test',
        is_distributed=distributed
    )
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("pysgg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
