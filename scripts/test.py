#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:39:57 2024

@author: fz
"""

from ultralytics import YOLO

import os
print(os.getcwd(), os.path.dirname(__file__))
# Load a model
# img_path = r"/mnt/18T/datasets/shudian_2025/images/test/"
img_path = r"/data1/datasets/tmp2"
# img_path = r"/data1/home/fz/desktop/ultralytics-v11/data/yuhang/"
output_path = r"./tmp_out"
os.makedirs(output_path, exist_ok=True)
model = YOLO("./Compare-yolov8s/yolov8x_ycx3/weights/best.pt")  # 
# model.val()
# raise
# import pdb;pdb.set_trace()
results = model.predict(source=img_path, save=True, conf=0.35)
# results = model.predict(source="/data1/home/fz/desktop/ultralytics-main/data", iou=0.5, save=False, conf=0.35)
# reference_t = {
#     0: 'fire', 
#     1: 'smoke', 
#     2: 'niaochao', 
#     3: 'dxyw', 
#     4: 'sgjx', 
#     5: 'caigangwa', 
#     6: 'fangchenwang', 
#     7: 'tadiao'
#     }

reference_t = {
    # 0: "pump_truck",
    0: "cement_mixer_truck",
    # 2: "piling_rig",
    # 3: "oil_tanker_truck",
    # 4: "truck_crane_b"
    # 0: "fire",
    # 1: "smoke",
    # 2: "nest",
    # 3: "dxyw",
    # 4: "coated_steel_sheet",
    # 5: "dustproof_net",
    # 6: "tower_crane",
    # 7: "truck_crane",
    # 8: "bulldozer",
    # 9: "pump_truck",
    # 10: "cement_mixer_truck",
    # 11: "truck",
    # 12: "excavator",
    # 13: "greenhouse",
    # 14: "tarpaulin",
    # 15: "piling_rig",
    # 16: "oil_tanker_truck",
    # 17: "truck_crane_b"
    }

for result in results:
    txt_path = os.path.join(output_path, os.path.basename(result.path)[:-4] + '.txt')
    # if not os.path.exists(txt_path):
    #     pass
    # else:
    #     txt_path = os.path.join(output_path, os.basename(result.path) + '.txt')
    with open(txt_path, "w") as f:
        for boxes in result.boxes:
            # print(result.path, box.cls, box.conf, box.xyxy)
            labels = boxes.cls.tolist()
            scores = boxes.conf.tolist()
            box = boxes.xyxy.tolist()
            for label, score, box in zip(labels, scores, box):
                x1, y1, x2, y2 = box
                result_str = f"{reference_t[int(label)]} {score:.3f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                # # 写入文件
                f.write(result_str)


# result_files = [os.path.splitext(f)[0] for f in os.listdir(output_path)]
# img_files = [os.path.splitext(f)[0] for f in os.listdir(img_path)]
# for img_file in os.listdir(img_files):
#     img_name, img_ext = os.path.splitext(img_file)
#     if img_name not in result_files:
#         with open()