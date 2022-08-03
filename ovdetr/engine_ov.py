# ------------------------------------------------------------------------
# OV DETR
# Copyright (c) S-LAB, Nanyang Technological University. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import math
import sys
from typing import Iterable

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.cuda.amp import GradScaler, autocast

import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from datasets.data_prefetcher import data_prefetcher


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    masks: bool = False,
    amp: bool = False,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    scaler = GradScaler()

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        with autocast(enabled=amp):
            if not masks:
                outputs = model(samples, targets)
            else:
                outputs = model(samples, targets, criterion)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model, criterion, postprocessors, data_loader, base_ds, device, output_dir, label_map, amp
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, data_loader.dataset.cat2label, label_map)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=amp):
            outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results, topk_boxes = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            outputs_masks = outputs["pred_masks"].squeeze(2)

            bs = len(topk_boxes)
            outputs_masks_new = [[] for _ in range(bs)]
            for b in range(bs):
                for index in topk_boxes[b]:
                    outputs_masks_new[b].append(outputs_masks[b : b + 1, index : index + 1, :, :])
            for b in range(bs):
                outputs_masks_new[b] = torch.cat(outputs_masks_new[b], 1)
            outputs["pred_masks"] = torch.cat(outputs_masks_new, 0)

            results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator


@torch.no_grad()
def lvis_evaluate(
    model, criterion, postprocessors, data_loader, base_ds, device, output_dir, label_map, amp
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    lvis_results = []

    cat2label = data_loader.dataset.cat2label
    label2cat = {v: k for k, v in cat2label.items()}

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=amp):
            outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results, topk_boxes = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            outputs_masks = outputs["pred_masks"].squeeze(2)

            bs = len(topk_boxes)
            outputs_masks_new = [[] for _ in range(bs)]
            for b in range(bs):
                for index in topk_boxes[b]:
                    outputs_masks_new[b].append(outputs_masks[b : b + 1, index : index + 1, :, :])
            for b in range(bs):
                outputs_masks_new[b] = torch.cat(outputs_masks_new[b], 1)
            outputs["pred_masks"] = torch.cat(outputs_masks_new, 0)

            results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

        for target, output in zip(targets, results):
            image_id = target["image_id"].item()

            if "masks" in output.keys():
                masks = output["masks"].data.cpu().numpy()
                masks = masks > 0.5
                rles = [
                    mask_util.encode(
                        np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                    )[0]
                    for mask in masks
                ]
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")

            boxes = convert_to_xywh(output["boxes"])
            for ind in range(len(output["scores"])):
                temp = {
                    "image_id": image_id,
                    "score": output["scores"][ind].item(),
                    "category_id": output["labels"][ind].item(),
                    "bbox": boxes[ind].tolist(),
                }
                if label_map:
                    temp["category_id"] = label2cat[temp["category_id"]]
                if "masks" in output.keys():
                    temp["segmentation"] = rles[ind]

                lvis_results.append(temp)

    rank = torch.distributed.get_rank()
    torch.save(lvis_results, output_dir + f"/pred_{rank}.pth")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    torch.distributed.barrier()
    if rank == 0:
        world_size = torch.distributed.get_world_size()
        for i in range(1, world_size):
            temp = torch.load(output_dir + f"/pred_{i}.pth")
            lvis_results += temp

        from lvis import LVISEval, LVISResults

        lvis_results = LVISResults(base_ds, lvis_results, max_dets=300)
        for iou_type in iou_types:
            lvis_eval = LVISEval(base_ds, lvis_results, iou_type)
            lvis_eval.run()
            lvis_eval.print_results()
    torch.distributed.barrier()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, None
