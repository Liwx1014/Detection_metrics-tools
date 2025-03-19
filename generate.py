from scripts.generate_xml_by_model import generate_xml
from scripts.generate_gt_by_xml import generate_gt
from scripts.generate_pred_by_yolo import generate_pred
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert output to xml, gt, pred')
    parser.add_argument('-c', '--config-path', type=str, required=False, default="./projects/12class.yaml", help='配置文件路径')
    parser.add_argument('-t', '--test-path', type=str, required=False, default="/data1/datasets/shudian_2025/images/test/", help='测试路径')
    parser.add_argument('-m', '--model', type=str, choices=['dfine', 'deim', 'yolo'], default="dfine", help='选择加载的模型: dfine/deim/yolo, gt模式的label_mapping与此有关')
    parser.add_argument('-mo', '--mode',type=str, choices=['xml', 'gt', 'pred'], default='gt', help='选择模式: xml/gt/pred')
    parser.add_argument('-mp', '--model-path', type=str, required=False, default="./model/yolov8/checkpoints/best.pt", help='模型加载路径')
    
    # generate xml from model
    parser.add_argument('-th', '--nms-thres', type=float, default=0.5, help='nms阈值: float/None, 分别代表固定阈值和默认的置信度字典') 
    parser.add_argument('--nms-class', default="all", help='nms过滤的类别: # None/all/[1,2,3], 分别代表不过滤/全过滤/过滤指定类别') 
    parser.add_argument('--xml-output', type=str, default=None, help='xml结果保存位置, 默认为 ./datasets/{project_name}/xmls')

    # generate gt from xml
    parser.add_argument('--gt-output', type=str, default=None, help='gt结果保存位置, 默认为 ./datasets/{project_name}/gt')

    # generate pred from yolo
    parser.add_argument('--pred-output', type=str, default=None, help='gt结果保存位置, 默认为 ./datasets/{project_name}/pred')
    parser.add_argument('--conf', type=float, default=0.35, help='yolo预测置信度阈值, 默认0.35')

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    if args.mode == "xml" or (args.mode == "pred" and (args.model == "dfine" or args.model == "deim")):
        generate_xml(args)
    elif args.mode == "gt":
        generate_gt(args)
    elif args.mode == "pred" and args.model == "yolo":
        generate_pred(args)

if __name__ == '__main__':
    main()
