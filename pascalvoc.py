import argparse
import glob
import os
import shutil
import sys
import json
import _init_paths

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat
import traceback


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

def compute_mAP(gtFolder, detFolder):
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
            if cl == "oil_tanker_truck":
                continue
            # if cl not in valClassNames:
            #     continue
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
        
    except:
        print('crash')
        print(traceback.format_exc())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算目标检测的mAP指标')
    parser.add_argument('-d', '--det', type=str, required=True, help='检测结果文件夹路径，包含txt格式的检测结果')
    parser.add_argument('-g', '--gt', type=str, required=True, help='真值标注文件夹路径，包含txt格式的标注文件')
    parser.add_argument('-i', '--iou', type=float, default=0.5, help='IOU阈值，默认为0.5')
    
    args = parser.parse_args()  
    
    if not os.path.exists(args.det):
        print(f"错误：检测结果文件夹 {args.det} 不存在")
        sys.exit(1)
    if not os.path.exists(args.gt):
        print(f"错误：真值标注文件夹 {args.gt} 不存在")
        sys.exit(1)
        
    compute_mAP(args.gt, args.det)