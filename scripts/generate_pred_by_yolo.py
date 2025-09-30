#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用YOLOv8模型生成预测结果
"""

import os
import sys
import argparse
import yaml
# from model.yolov8.ultralytics import YOLO
project_root = os.path.dirname(os.path.abspath(__file__))
existing_path = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = project_root + os.pathsep + existing_path
from ultralytics import YOLO
from tqdm import tqdm
import time
def generate_pred(args):
    # 读取配置文件
    config_path = args.config_path
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取项目名称和图片路径
    project_name = config.get('project_name', 'default')
    image_folder = config['paths']['image_folder']
    date_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # 设置输出路径
    output_path = args.pred_output if args.pred_output else os.path.join('./datasets', project_name, args.model + "_pred_12x_wbf")
    os.makedirs(output_path, exist_ok=True)
    
    model_path = args.model_path if args.model_path else config['model_path'] 
    print(f"model path is {model_path}")
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在")
        sys.exit(1)
    
    # 加载模型
    start_time = time.time()
    model = YOLO(model_path)
    
    # 获取标签映射
    label_mapping = {str(k): v for k, v in config['yolo_mapping'].items()}
    class_mapping = {str(k): v for k, v in config['class_mapping'].items()}
    if 'valid_list' in config and config['valid_list']:
        valid_classes = config['valid_list']
    else:
        valid_classes = set(label_mapping.values())
    
    # 运行预测
    print(f"正在处理图片目录: {image_folder}")
    results = model.predict(source=image_folder, save=False, stream=True, rect=False, verbose=False)
    
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
                    #print(class_name)
                    class_name = class_mapping[class_name] if class_name in class_mapping.keys() else class_name # 将yolo输出的name，再做合并转换
                    if class_name in valid_classes:
                        result_str = f"{class_name} {score:.3f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                        f.write(result_str)
    end_time = time.time()
    print(f"预测完成，耗时: {end_time - start_time:.4f} 秒")
 

if __name__ == "__main__":
    main()