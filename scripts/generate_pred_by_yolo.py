#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用YOLOv8模型生成预测结果
"""

import os
import sys
import argparse
import yaml
from ultralytics import YOLO
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='使用YOLOv8模型生成预测结果')
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件路径，例如：projects/12class.yaml')
    parser.add_argument('-m', '--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('-o', '--output', type=str, help='输出文件夹路径，默认为 ./datasets/{project_name}/pred')
    parser.add_argument('--conf', type=float, default=0.35, help='置信度阈值，默认0.35')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取项目名称和图片路径
    project_name = config.get('project_name', 'default')
    image_folder = config['paths']['image_folder']
    
    # 设置输出路径
    output_path = args.output if args.output else os.path.join('./datasets', project_name, 'pred')
    os.makedirs(output_path, exist_ok=True)
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误：模型文件 {args.model} 不存在")
        sys.exit(1)
    
    # 加载模型
    model = YOLO(args.model)
    
    # 获取标签映射
    label_mapping = {str(k): v for k, v in config['label_mapping'].items()}
    
    # 运行预测
    print(f"正在处理图片目录: {image_folder}")
    results = model.predict(source=image_folder, save=False, conf=args.conf)
    
    # 保存结果
    print(f"正在保存预测结果到: {output_path}")
    for result in tqdm(results, desc="Saving predictions"):
        txt_path = os.path.join(output_path, os.path.basename(result.path)[:-4] + '.txt')
        with open(txt_path, "w", encoding='utf-8') as f:
            for boxes in result.boxes:
                labels = boxes.cls.tolist()
                scores = boxes.conf.tolist()
                box = boxes.xyxy.tolist()
                for label, score, box in zip(labels, scores, box):
                    x1, y1, x2, y2 = box
                    # 使用label_mapping获取类别名称
                    class_name = label_mapping[str(int(label))]
                    result_str = f"{class_name} {score:.3f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                    f.write(result_str)
    
    print("预测完成！")

if __name__ == "__main__":
    main()