#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:18:09 2024

@author: fz
"""

import traceback
import sys, os, time
import requests
from torchvision.ops import batched_nms
from typing import List, Optional, Union, Tuple
import numpy as np 
import cv2
import base64
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
import logging
import math
import io
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml


def non_max_suppression(
    predicted_labels: torch.Tensor,
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    label_mapping: Union[dict],
    conf_thres: Union[float, dict] = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[Union[List[int], torch.Tensor, str]] = None,
    agnostic: bool = False,
    max_det: int = 300,
    max_nms: int = 30000,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    自定义非极大抑制（NMS）函数，适用于新模型的输出格式。

    参数：
        predicted_labels (torch.Tensor): [batch_size, num_boxes] 每个框的类别标签
        predicted_boxes (torch.Tensor): [batch_size, num_boxes, 4] 每个框的坐标 [x1, y1, x2, y2]
        predicted_scores (torch.Tensor): [batch_size, num_boxes] 每个框的置信度得分
        conf_thres (float 或 dict): 置信度阈值，低于此值的框将被过滤。如果是一个字典，键为类别，值为对应的置信度阈值。
        iou_thres (float): IoU 阈值，用于NMS，重叠度高于此值的框将被抑制
        classes (List[int] 或 torch.Tensor 或 str, optional): 指定需要过滤的类别索引。如果为 None，不过滤任何类别。如果为 "all"，则过滤所有类别。
        agnostic (bool): 是否忽略类别进行NMS
        max_det (int): 每张图片最多保留的检测框数量
        max_nms (int): NMS前最多处理的预测框数量

    返回：
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            - List of filtered boxes for each image in the batch, shape [num_selected_boxes, 4]
            - List of filtered labels for each image in the batch, shape [num_selected_boxes]
            - List of filtered scores for each image in the batch, shape [num_selected_boxes]
    """

    # 获取批量大小
    batch_size: int = predicted_boxes.shape[0]
    # 初始化输出列表，用于存储每张图片的过滤后结果
    output_boxes: List[torch.Tensor] = []
    output_labels: List[torch.Tensor] = []
    output_scores: List[torch.Tensor] = []

    # 设定NMS的时间限制，防止处理时间过长
    time_limit: float = 2.0 + 0.05 * batch_size  # 以秒为单位
    t_start: float = time.time()  # 记录开始时间

    # 遍历每一张图片
    for i in range(batch_size):
        # 提取当前图片的预测框、得分和标签
        boxes: torch.Tensor = predicted_boxes[i]          # [num_boxes, 4]
        scores: torch.Tensor = predicted_scores[i]        # [num_boxes]
        labels: torch.Tensor = predicted_labels[i]        # [num_boxes]
        # 初始化 mask
        label_mask = (labels > 0) & (labels <= len(label_mapping))

        # 应用掩码过滤掉标签值大于label_mapping长度的框
        boxes = boxes[label_mask]
        scores = scores[label_mask]
        labels = labels[label_mask]

        mask = torch.ones_like(scores, dtype=torch.bool)  # 初始化为全 True
        # 根据 conf_thres 和 classes 的不同组合进行过滤
        if isinstance(conf_thres, dict) and isinstance(classes, list):
            # 如果 conf_thres 是字典且 classes 是列表或张量
            classes = torch.tensor(classes, device=labels.device)
            mask_classes = (labels.unsqueeze(1) == classes).any(1)
            # 对每个类别应用特定的置信度阈值
            for label in classes:
                label_idx = labels == int(label)
                threshold = conf_thres.get(label_mapping[int(label)], 0.0)
                mask[label_idx] = (scores[label_idx] > threshold)
            # 应用类别掩码
            mask = mask & mask_classes
        elif isinstance(conf_thres, float) and isinstance(classes, list):
            # 如果 conf_thres 是浮点数且 classes 是列表或张量
            classes = torch.tensor(classes, device=labels.device)
            mask_classes = (labels.unsqueeze(1) == classes).any(1)
            # 应用统一的置信度阈值
            mask_scores = scores > conf_thres
            # 应用类别掩码和置信度掩码
            mask = mask & mask_classes & mask_scores
        elif classes is None:
            # 如果 classes 是 None，不过滤任何类别
            pass
        elif classes == "all":
            # 如果 classes 是 "all"，过滤所有类别
            if isinstance(conf_thres, dict):
                # 如果 conf_thres 是字典，对每个类别应用特定的置信度阈值
                for label in torch.unique(labels):
                    label_idx = labels == int(label)
                    threshold = conf_thres.get(label_mapping[int(label)], 0.0)
                    mask[label_idx] = (scores[label_idx] > threshold)
            else:
                # 如果 conf_thres 是浮点数，应用统一的置信度阈值
                mask = scores > conf_thres

        # 应用 mask 来过滤预测框、得分和标签
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # 如果过滤后没有剩余的预测框，添加空的结果并继续
        if boxes.numel() == 0:
            output_boxes.append(torch.empty((0, 4), device=boxes.device))
            output_scores.append(torch.empty((0,), device=boxes.device))
            output_labels.append(torch.empty((0,), dtype=torch.int64, device=boxes.device))
            continue

        if boxes.shape[0] > max_nms:
            # 保留置信度最高的 max_nms 个预测框
            scores, idx = scores.topk(max_nms)
            boxes = boxes[idx]
            labels = labels[idx]

        # 如果 agnostic=True，则忽略类别信息，所有预测框视为同一类别
        if agnostic:
            nms_labels: torch.Tensor = torch.zeros_like(labels)
        else:
            nms_labels = labels

        # 执行 NMS，返回保留的框的索引
        keep: torch.Tensor = batched_nms(boxes, scores, labels, iou_thres)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        # int_boxes = boxes.to(torch.int)
        # unique_boxes, inverse_indices = torch.unique(int_boxes, dim=0, return_inverse=True)
        # max_score_indices = []
        # for i in range(unique_boxes.shape[0]):
        #     # 获取当前组的索引
        #     group_indices = torch.where(inverse_indices == i)[0]
        #     # 找到分数最高的索引
        #     max_score_idx = group_indices[torch.argmax(scores[group_indices])]
        #     max_score_indices.append(max_score_idx)
        # max_score_indices.sort()
        # keep = torch.tensor(max_score_indices, dtype=torch.long, device=keep.device)


        # 根据保留的索引提取最终的预测框、得分和标签

        nms_labels: torch.Tensor = torch.zeros_like(labels)
        keep: torch.Tensor = batched_nms(boxes, scores, nms_labels, 0.95)

        if keep.shape[0] > max_det:
            keep = keep[:max_det]
        
        selected_boxes: torch.Tensor = boxes[keep]
        selected_scores: torch.Tensor = scores[keep]
        selected_labels: torch.Tensor = labels[keep]    
        output_boxes.append(selected_boxes)
        output_scores.append(selected_scores)
        output_labels.append(selected_labels)

        if (time.time() - t_start) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    # 返回过滤后的预测框、标签和得分
    return output_boxes, output_labels, output_scores

def detect(model, im_pil, label_mapping, cls_score_thre, cls_need_merge, device):
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(im_data, orig_size)
    predicted_labels, predicted_boxes, predicted_scores = output # (x1, y1, x2, y2)
    # predicted_labels, predicted_boxes, predicted_scores = filter_boxes(predicted_labels, predicted_boxes, predicted_scores)
    print(label_mapping, cls_score_thre, cls_need_merge)
    filtered_boxes, filtered_labels, filtered_scores = non_max_suppression(
        predicted_labels,
        predicted_boxes,
        predicted_scores,
        label_mapping=label_mapping,
        conf_thres=0.35,
        iou_thres=0.5,
        classes=cls_need_merge,
        agnostic=False,
        max_det=100,
        max_nms=1000
    )

    print(filtered_boxes[0].shape,filtered_labels[0].shape,filtered_scores[0].shape)
    return filtered_labels[0].cpu().numpy(), filtered_boxes[0].cpu().numpy(), filtered_scores[0].cpu().numpy()

def output_to_xml(labels, boxes, scores, img_size, label_mapping, class_mapping, img_name, img_path, save_path):
    # Create root element
    root = ET.Element("annotation")

    # Add folder element
    folder = ET.SubElement(root, "folder")
    folder.text = "images"

    # Add filename element
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(img_name)

    # Add path element
    path = ET.SubElement(root, "path")
    path.text = img_path

    # Add source element
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    # Add size element
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_size[0])
    height = ET.SubElement(size, "height")
    height.text = str(img_size[1])
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    # Add segmented element
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    # Add object elements
    for label, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(int(x2), img_size[0]), min(int(y2), img_size[1])
        # Create object element
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")
        class_name = label_mapping[int(label)]
        # if class_name in class_mapping:
        #     class_name = class_mapping[class_name]
        name.text = class_name
        truncated = ET.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object, "difficult")
        difficult.text = "0"

        # Create bndbox element
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(x1))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(y1))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(x2))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(y2))

    # Create XML file
    xml_path = os.path.join(save_path, os.path.splitext(img_name)[0] + '.xml')
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    with open(xml_path, 'w', encoding='utf-8') as xml_file:
        xml_file.write(pretty_xml)

def compute_iou(box1, box2):
    """
    计算两个框的 IoU。
    参数:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    返回:
        IoU 值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union != 0 else 0

def read_repeat_info(repeat_txt_path):
    """
    读取 repeat.txt 文件，返回每个文件名对应的重复框信息。
    返回:
        repeat_info: dict[file_name] = [(category, x1, y1, x2, y2), ...]
    """
    repeat_info = {}
    with open(repeat_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[0]
            category, x1, y1, x2, y2 = parts[1], float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if file_name not in repeat_info:
                repeat_info[file_name] = []
            repeat_info[file_name].append((category, x1, y1, x2, y2))
    return repeat_info

def output_to_pred(labels, boxes, scores, img_size, label_mapping, class_mapping, img_name, save_path):
    with open(os.path.join(save_path, os.path.splitext(img_name)[0] + ".txt"), "w") as f:
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(int(x2), img_size[0]), min(int(y2), img_size[1])
            # 格式化字符串
            class_name = label_mapping[int(label)]
            if class_name in class_mapping:
                class_name = class_mapping[class_name]
            if class_name is not None:
                result_str = f"{class_name} {score:.3f} {x1} {y1} {x2} {y2}\n"
                # 写入文件
                f.write(result_str)

def process_image(args, img_name, img_paths, model, label_mapping, class_mapping, cls_score_thre, cls_need_merge, project_name, device):
    """
    处理单张图像的函数。
    """
    try:
        img_path = os.path.join(img_paths, img_name)
        im_pil = Image.open(img_path).convert('RGB')
        labels, boxes, scores = detect(model, im_pil, label_mapping, cls_score_thre, cls_need_merge, device)
        if args.mode == "xml":
            save_path = args.xml_output if args.xml_output else os.path.join('./datasets', project_name, args.model + 'xmls')
            os.makedirs(save_path, exist_ok=True)
            output_to_xml(labels, boxes, scores, im_pil.size, label_mapping, class_mapping, img_name, img_path, save_path)
        else:
            save_path = args.pred_output if args.pred_output else os.path.join('./datasets', project_name, args.model + '_pred')
            os.makedirs(save_path, exist_ok=True)
            output_to_pred(labels, boxes, scores, im_pil.size, label_mapping, class_mapping, img_name, save_path)
        return None  # 表示成功
    except Exception as e:
        return (img_path, str(e))  # 返回错误信息
    
def generate_xml(args):
    config_path = args.config_path
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    img_paths = args.test_path
    if args.model == 'deim':
        from model.deim.core.engine.core import YAMLConfig
    else:
        if args.model != 'dfine':
            args.model = 'dfine'
            print("预设的模型不是deim或者dfine, 改为dfine")
        from model.dfine.core.src.core import YAMLConfig
    project_name = config.get('project_name', 'default')
    model_path = args.model_path if args.model_path else config['model_path'] 
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在")
        sys.exit(1)
    cls_score_thre = args.nms_thres if args.nms_thres else config['cls_score_thre'] 
    cls_need_merge = args.nms_class if args.nms_class else config['cls_need_merge']
    model_config = config['model_config'] 
    label_mapping = config['transform_mapping']
    class_mapping = config['class_mapping']
    device = "cuda:0"
    # device = "cuda:7"
    cfg = YAMLConfig(model_config)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    checkpoint = torch.load(model_path, map_location='cpu')
    state = checkpoint.get('ema', checkpoint['model'])['module']
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
    
        def forward(self, images, orig_target_sizes):
            with torch.no_grad():
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)
        
    model = Model().to(device)
    # logger.info(f"device: {next(model.parameters()).device}")
    
    img_names = os.listdir(img_paths)
    # from concurrent.futures import ThreadPoolExecutor, as_completed
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     # 创建一个 future 到 img_name 的字典
    #     futures = []
    #     for img_name in img_names:
    #         future = executor.submit(
    #             process_image,
    #             args,
    #             img_name,
    #             img_paths,
    #             model,
    #             label_mapping,
    #             class_mapping,
    #             cls_score_thre,
    #             cls_need_merge,
    #             project_name,
    #             device
    #         )
    #         futures.append(future)
    #     error_txt = os.path.join('./datasets', project_name, args.model + "_error_log.txt")  # 记录错误的文件路径
    #     # 使用 tqdm 显示进度条
    #     for future in tqdm(as_completed(futures), total=len(img_names), desc="Processing Images"):
    #         result = future.result()  # 获取执行结果
    #         if result:  # 如果 result 不为空，表示发生了错误
    #             error_path, error = result
    #             with open(error_txt, 'a') as error_log:
    #                 error_log.write(f'{error_path} 生成时发生错误: {error}\n')
    for img_name in tqdm(img_names, desc="Processing Images"):
        if img_name.endswith('.npy'):
            continue
        img_path = os.path.join(img_paths, img_name)
        im_pil = Image.open(img_path).convert('RGB')
        cls_score_thre = 0.25 #和yolo对齐
        labels, boxes, scores = detect(model, im_pil, label_mapping, cls_score_thre, cls_need_merge, device)
        if args.mode == "xml":
            save_path = args.xml_output if args.xml_output else os.path.join('./datasets', project_name, 'xmls')
            os.makedirs(save_path, exist_ok=True)
            output_to_xml(labels, boxes, scores, im_pil.size, label_mapping, class_mapping, img_name, img_path, save_path)
        else:
            save_path = args.pred_output if args.pred_output else os.path.join('./datasets', project_name, args.model + '_pred')
            os.makedirs(save_path, exist_ok=True)
            output_to_pred(labels, boxes, scores, im_pil.size, label_mapping, class_mapping, img_name, save_path)

if __name__ == '__main__':
    
    # python generate_xml.py -c ./config.yaml -i /data1/datasets/shudian_2025/images/test/ -t 0.25

    # logger_file = r"./loggers"
    # if not os.path.exists(logger_file):
    #     os.makedirs(logger_file, exist_ok=True)
    # logging.basicConfig(
    #     level=logging.INFO,  # 设置日志级别
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
    #     handlers=[
    #         logging.FileHandler("./loggers/test.log"),  # 将日志写入文件
    #         logging.StreamHandler()  # 同时将日志输出到控制台
    #     ]
    # )
    # logger = logging.getLogger(__name__)

    args = parse_args()
    main(args)