import argparse
import glob
import os
import shutil
import sys
import json
import yaml
import _init_paths
import cv2
import numpy as np
from collections import defaultdict
import io  # 确保导入io模块

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat
import traceback

def visualize_box(image, box, color, label, thickness=2):
    """在图片上绘制边界框和标签"""
    x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    # 添加黑色背景使文字更清晰
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
    cv2.rectangle(image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
    cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness)
    return image

def analyze_errors(allBoundingBoxes, image_folder, target_class, output_folder, xml_folder, max_samples=5000):
    """对指定类别进行错误分析并可视化"""
    # 创建输出目录
    error_dir = os.path.join(output_folder, f"{target_class}_error_analysis")
    os.makedirs(error_dir, exist_ok=True)
    
    # 创建结构化数据保存目录
    structured_data_dir = os.path.join(output_folder, "structured_data")
    os.makedirs(structured_data_dir, exist_ok=True)
    
    # 创建评估器并进行匹配
    evaluator = Evaluator()
    
    # 收集错误样本
    error_samples = defaultdict(lambda: {'fp': [], 'fn': [], 'tp': [], 'gt': []})  # 图片名 -> 各类框
    
    # 获取所有检测结果和真值标注
    detections = []  # [imageName, class, confidence, bbox]
    groundtruths = []  # [imageName, class, confidence=1, bbox]
    
    for bb in allBoundingBoxes.getBoundingBoxes():
        if bb.getClassId() != target_class:
            continue
            
        img_name = bb.getImageName()
        bbox = bb.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        
        if bb.getBBType() == BBType.Detected:
            detections.append([img_name, target_class, bb.getConfidence(), bbox])
        else:  # Ground Truth
            groundtruths.append([img_name, target_class, 1, bbox])
            error_samples[img_name]['gt'].append(bbox)
    
    # 按置信度降序排序检测结果
    detections.sort(key=lambda x: x[2], reverse=True)
    
    # 创建GT字典，key为图片名
    gts = defaultdict(list)
    for gt in groundtruths:
        gts[gt[0]].append(gt)
    
    # 标记每个检测结果
    for det in detections:
        img_name = det[0]
        det_bbox = det[3]
        conf = det[2]
        
        if img_name not in gts:
            # 如果图片中没有GT，则所有检测都是FP
            error_samples[img_name]['fp'].append((det_bbox, conf))
            continue
            
        # 计算与所有GT的IOU
        gt_boxes = [gt[3] for gt in gts[img_name]]
        ious = [evaluator.iou(det_bbox, gt_box) for gt_box in gt_boxes]
        
        if not ious or max(ious) < 0.5:  # IOU阈值设为0.5
            # 没有匹配的GT，是FP
            error_samples[img_name]['fp'].append((det_bbox, conf))
        else:
            # 是正确检测（TP）
            error_samples[img_name]['tp'].append((det_bbox, conf))
    
    # 找出未被检测到的GT（FN）
    for img_name, gt_list in gts.items():
        for gt in gt_list:
            gt_bbox = gt[3]
            # 计算与所有检测结果的IOU
            det_boxes = [d[3] for d in detections if d[0] == img_name]
            ious = [evaluator.iou(gt_bbox, det_box) for det_box in det_boxes]
            
            if not ious or max(ious) < 0.5:  # IOU阈值设为0.5
                # 没有匹配的检测结果，是FN
                error_samples[img_name]['fn'].append(gt_bbox)
    
    # 保存可视化结果
    print(f"\n分析类别 '{target_class}' 的错误样本:")
    
    # 统计包含错误的图片数量
    fp_images = sum(1 for samples in error_samples.values() if samples['fp'])
    fn_images = sum(1 for samples in error_samples.values() if samples['fn'])
    print(f"发现 {fp_images} 张图片包含误检(FP)样本")
    print(f"发现 {fn_images} 张图片包含漏检(FN)样本")
    
    # 处理所有样本
    processed = 0
    structured_data = []  # 用于保存结构化数据
    for img_name, samples in error_samples.items():
        # 只处理包含错误的图片
        if not (samples['fp'] or samples['fn']):
            continue
            
        # 添加结构化数据
        structured_data.append({
            "image_name": img_name,
            "image_path": os.path.join(image_folder, img_name + get_image_extension(image_folder, img_name)),
            "fp": [{"bbox": box, "confidence": conf, "save": False} for box, conf in samples['fp']],
            "fn": [{"bbox": box, "save": True} for box in samples['fn']],
            "original_xml_path": os.path.join(xml_folder, img_name + '.xml'),  # 使用 xml_folder 而不是 image_folder
            "class_name": target_class  # 添加 class_name 字段
        })
        
        # 仅在未达到 max_samples 时保存可视化图片
        if processed < max_samples:
            img_path = os.path.join(image_folder, img_name + get_image_extension(image_folder, img_name))
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 先绘制GT框（绿色虚线）
            for box in samples['gt']:
                img = visualize_box(img, box, (0, 255, 0), "GT", 1)
                
            # 绘制TP框（绿色实线）
            for box, conf in samples['tp']:
                img = visualize_box(img, box, (0, 255, 0), f"TP:{conf:.2f}", 2)
                
            # 绘制FP框（红色）
            for box, conf in samples['fp']:
                img = visualize_box(img, box, (0, 0, 255), f"FP:{conf:.2f}", 2)
                
            # 绘制FN框（蓝色）
            for box in samples['fn']:
                img = visualize_box(img, box, (255, 0, 0), "FN", 2)
                
            # 在图片上添加错误统计信息
            info_text = f"FP:{len(samples['fp'])} FN:{len(samples['fn'])} TP:{len(samples['tp'])}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(os.path.join(error_dir, f"{img_name}.jpg"), img)
            processed += 1
    
    # 将结构化数据保存到文件
    structured_data_path = os.path.join(structured_data_dir, f"{target_class}_error_analysis.json")
    with io.open(structured_data_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=4)
    
    print(f"已保存 {processed} 张错误分析结果到 {error_dir}")
    print(f"结构化数据已保存到 {structured_data_path}")
    print(f"图例说明：")
    print("- 绿色虚线框：真值标注(GT)")
    print("- 绿色实线框：正确检测(TP)")
    print("- 红色框：误检(FP)")
    print("- 蓝色框：漏检(FN)")

def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     refFolder=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    files = os.listdir(directory)
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    
    for f in files:
        if not f.endswith('.txt'): continue
        if not isGT and not f in os.listdir(refFolder): continue
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(directory, f), "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.GroundTruth,
                                 format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.Detected,
                                 confidence,
                                 format=bbFormat)
            # print(idClass, x, y, w, h)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses

def getBoundingBoxesSingle(f,
                           isGT,
                           bbFormat,
                           coordType,
                           allBoundingBoxes=None,
                           allClasses=None,
                           imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    nameOfImage = f.split('/')[-1].replace(".txt", "")
    if not os.path.exists(f):
        return False, False
    fh1 = open(f, "r")
    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        if isGT:
            # idClass = int(splitLine[0]) #class
            idClass = (splitLine[0])  # class
            x = float(splitLine[1])
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(nameOfImage,
                                idClass,
                                x,
                                y,
                                w,
                                h,
                                coordType,
                                imgSize,
                                BBType.GroundTruth,
                                format=bbFormat)
        else:
            # idClass = int(splitLine[0]) #class
            idClass = (splitLine[0])  # class
            confidence = float(splitLine[1])
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(nameOfImage,
                                idClass,
                                x,
                                y,
                                w,
                                h,
                                coordType,
                                imgSize,
                                BBType.Detected,
                                confidence,
                                format=bbFormat)
        # print(idClass, x, y, w, h)
        allBoundingBoxes.addBoundingBox(bb)
        if idClass not in allClasses:
            allClasses.append(idClass)
    fh1.close()
    return allBoundingBoxes, allClasses

def compute_mAP(gtFolder, detFolder, deep_analysis=None, image_folder=None, output_folder=None, xml_folder=None):
    # Get current path to set default folders
    try:
        currentPath = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists(gtFolder):
            os.makedirs(gtFolder)
        gtFormat = BBFormat.XYX2Y2
        gtCoordType = CoordinatesType.Absolute
        detFormat = BBFormat.XYX2Y2
        detCoordType = CoordinatesType.Absolute
        imgSize = (0, 0)
        iouThreshold = 0.5
        # Get groundtruth boxes
        allBoundingBoxes, allClasses = getBoundingBoxes(gtFolder,
                                                        True,
                                                        gtFormat,
                                                        gtCoordType,
                                                        imgSize=imgSize,
                                                        refFolder=gtFolder)
        # Get detected boxes
        allBoundingBoxes, allClasses = getBoundingBoxes(detFolder,
                                                        False,
                                                        detFormat,
                                                        detCoordType,
                                                        allBoundingBoxes,
                                                        allClasses,
                                                        imgSize=imgSize,
                                                        refFolder=gtFolder)
        allClasses.sort()
        evaluator = Evaluator()
        acc_AP = 0
        validClasses = 0

        # Plot Precision x Recall curve
        detections = evaluator.PlotPrecisionRecallCurve(
            allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iouThreshold,  # IOU threshold
            method=MethodAveragePrecision.YOLOAP,
            showAP=True,  # Show Average Precision in the title of the plot
            showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
            savePath=None,
            showGraphic=False)

        precision_sum = 0
        recall_sum = 0
        # each detection is a class
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Class", "TP", "FP", "total positives", "FPR", "Recall", "AP"]
        # print('cls', 'print')
        # valClassNames = ['niaochao', 'sgjx', 'tadiao']
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            TP = metricsPerClass['total TP']
            FP = metricsPerClass['total FP']
            try:    
                # print(cl, '%.4f' % recall[-1], '%.4f' % (1.0 - precision[-1]), '%.4f' % ap)
                table.add_row([cl, TP, FP, totalPositives, '%.4f' % (1.0 - precision[-1]), '%.4f' % recall[-1], '%.4f' % ap])
            except Exception as e:
                print(e)
                continue
            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                if len(precision) > 0:
                    precision_sum += precision[-1]
                if len(recall) > 0:
                    recall_sum += recall[-1]

        mAP = acc_AP / validClasses
        precision_ave = precision_sum / validClasses
        recall_ave = recall_sum / validClasses
        mAP_str = "{0:.2f}%".format(mAP * 100)
        prec_str = "{0:.2f}%".format(precision_ave * 100)
        rec_str = "{0:.2f}%".format(recall_ave * 100)
        # print('mAP: %s' % mAP_str)
        # print('precision: %s' % prec_str)
        # print('recall: %s' % rec_str)
        table.add_row([" " for _ in table.field_names])
        table.add_row(["mAP"] + ['-'] * (len(table.field_names) - 2) + ['{0:.2f}%'.format(mAP * 100)])
        table.add_row(["precision"] + ['-'] * (len(table.field_names) - 2) + ['{0:.2f}%'.format(precision_ave * 100)])
        table.add_row(["recall"]+ ['-'] * (len(table.field_names) - 2) + ['{0:.2f}%'.format(recall_ave * 100)])
        print(table)
        
        # 如果开启深度分析，对指定类别进行分析
        if deep_analysis and image_folder and output_folder:
            if xml_folder is None:
                print("错误：xml_folder 未在配置文件中设置")
                sys.exit(1)
            if len(deep_analysis)>1:
                for target_class in deep_analysis:
                    assert target_class in allClasses, f"错误：类别 '{target_class}' 不存在于检测结果中"
                    analyze_errors(allBoundingBoxes, image_folder, target_class, output_folder, xml_folder)
            elif deep_analysis[0] == "all":
                for target_class in allClasses:
                    analyze_errors(allBoundingBoxes, image_folder, target_class, output_folder, xml_folder)
            else:
                analyze_errors(allBoundingBoxes, image_folder, deep_analysis[0], output_folder, xml_folder)
        
    except:
        print(traceback.format_exc())

def get_image_extension(folder_path, image_name):
    """
    查找给定图片名称的实际扩展名
    支持常见图片格式：jpg, jpeg, png, bmp, etc.
    """
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        if os.path.exists(os.path.join(folder_path, image_name + ext)):
            return ext
    return '.jpg'  # 如果都没找到，返回默认值

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算目标检测的mAP指标, python pascalvoc.py -g datasets/yiwu_6cls/gt -p datasets/yiwu_6cls/deim_pred  -c projects/yiwu_6cls.yaml --deep-analysis tower_crane')
    parser.add_argument('-p', '--pred', type=str, required=True, help='检测结果文件夹路径，包含txt格式的检测结果')
    parser.add_argument('-g', '--gt', type=str, help='真值标注文件夹路径，包含txt格式的标注文件')
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件路径，例如：projects/12class.yaml')
    parser.add_argument('-i', '--iou', type=float, default=0.5, help='IOU阈值，默认为0.5')
    parser.add_argument('-d', '--deep-analysis', type=str,nargs='+', help='指定要深度分析的类别名称，可以使用all分析所有类别')
    parser.add_argument('-v', '--vis-dir', type=str, help='可视化结果保存目录，默认为 ./datasets/{project_name}/analysis')
    
    args = parser.parse_args()

    # 加载项目配置文件
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取项目名称和图片路径
    project_name = config.get('project_name', 'default')
    image_folder = config['paths']['image_folder'] if args.deep_analysis else None
    xml_folder = config['paths'].get('xml_folder')  # 确保从 paths 中获取 xml_folder
    
    # 设置路径
    gt_folder = args.gt if args.gt else os.path.join('./datasets', project_name, 'gt')
    vis_dir = args.vis_dir if args.vis_dir else os.path.join('./datasets', project_name, 'analysis')
    
    if args.deep_analysis:
        if not os.path.exists(image_folder):
            print(f"错误：图片目录 {image_folder} 不存在")
            sys.exit(1)
        os.makedirs(vis_dir, exist_ok=True)
    
    if not os.path.exists(args.pred):
        print(f"错误：检测结果文件夹 {args.pred} 不存在")
        sys.exit(1)
    if not os.path.exists(gt_folder):
        print(f"错误：真值标注文件夹 {gt_folder} 不存在")
        sys.exit(1)
        
    compute_mAP(gt_folder, args.pred, args.deep_analysis, image_folder, vis_dir, xml_folder)