"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import mindspore #rxz
import torch.amp #找不到

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


# def train_one_epoch(model: mindspore.nn.Cell, criterion: mindspore.nn.Cell,
#                     data_loader: Iterable, optimizer: mindspore.nn.Optimizer,                     #rxz
#                     device, epoch: int, max_norm: float = 0, **kwargs):  #device： torch.device mindspore无需指定设备
#     model.train()
#     criterion.train()
#     metric_logger = MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = kwargs.get('print_freq', 10)
    
#     ema = kwargs.get('ema', None)
#     scaler = kwargs.get('scaler', None)

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         if scaler is not None:
#             with torch.autocast(device_type=str(device), cache_enabled=True):  #找不到
#                 outputs = model(samples, targets)
            
#             with torch.autocast(device_type=str(device), enabled=False):  #找不到
#                 loss_dict = criterion(outputs, targets)

#             loss = sum(loss_dict.values())
#             scaler.scale(loss).backward()
            
#             if max_norm > 0:
#                 scaler.unscale_(optimizer)
#                 mindspore.ops.clip_by_norm(model.parameters(), max_norm)  #rxz model..parameters这个参数有问题

#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()

#         else:
#             outputs = model(samples, targets)
#             loss_dict = criterion(outputs, targets)
            
#             loss = sum(loss_dict.values())
#             optimizer.zero_grad()
#             loss.backward()
            
#             if max_norm > 0: 
#                 mindspore.ops.clip_by_norm(model.parameters(), max_norm)  #rxz model..parameters这个参数有问题

#             optimizer.step()
        
#         # ema 
#         if ema is not None:
#             ema.update(model)

#         loss_dict_reduced = reduce_dict(loss_dict)
#         loss_value = sum(loss_dict_reduced.values())

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)

#         metric_logger.update(loss=loss_value, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
def train_one_epoch(
    model: nn.Cell,
    criterion: nn.Cell,
    data_loader: Iterable,
    optimizer: nn.Optimizer,
    epoch: int,
    max_norm: float = 0,
    **kwargs
):
    """
    训练一个 epoch 的函数。
    
    参数:
    - model (nn.Cell): 模型实例。
    - criterion (nn.Cell): 损失函数实例。
    - data_loader (Iterable): 数据加载器，提供批量数据。
    - optimizer (nn.Optimizer): 优化器实例。
    - epoch (int): 当前的 epoch 编号。
    - max_norm (float): 梯度裁剪的最大范数，默认为 0 表示不进行梯度裁剪。
    - **kwargs: 其他可选参数。
    """
    # 设置模型为训练模式
    model.set_train(True)
    criterion.set_train(True)

    # 初始化日志记录器
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    # 构建带有混合精度的训练网络
    if scaler is not None:
        net_with_criterion = WithLossCell(model, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer, scale_sense=scaler)
    else:
        train_network = WithLossCell(model, criterion)

    # 自定义训练步骤
    def train_step(samples, targets):
        outputs = train_network(samples, targets)
        return outputs

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 将输入数据转换为 Tensor（如果需要）
        samples = Tensor(samples)
        targets = [{k: Tensor(v) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with ms.ops.autocast(use_gpu=True):  # 使用 GPU 进行混合精度训练
                outputs = train_network(samples, targets)

            # 计算损失
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())

            # 反向传播并更新参数
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                # 对梯度进行裁剪
                grads = optimizer.parameters
                clipped_grads, _ = clip_by_global_norm(grads, max_norm)
                optimizer.set_parameters(clipped_grads)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = train_network(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                # 对梯度进行裁剪
                grads = optimizer.parameters
                clipped_grads, _ = clip_by_global_norm(grads, max_norm)
                optimizer.set_parameters(clipped_grads)

            optimizer.step()

        # 更新 EMA（如果启用）
        if ema is not None:
            ema.update(model)

        # 记录损失
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.learning_rate.asnumpy().item())

    # 同步日志记录
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: mindspore.nn.Cell, criterion: mindspore.nn.Cell, postprocessors, data_loader, base_ds, device, output_dir):  #rxz
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = mindspore.ops.Stack([t["orig_size"] for t in targets], dim=0)  #rxz
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator



