from scripts.generate_xml_by_model import generate_xml
from scripts.generate_gt_by_xml import generate_gt
from scripts.generate_pred_by_yolo import generate_pred
from scripts.yolo_2_voc import  convert_yolo_2_xml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert output to xml, gt, pred')
    parser.add_argument('-c', '--config-path', type=str, required=False, default="./projects/smoke.yaml", help='配置文件路径')
    parser.add_argument('-t', '--test-path', type=str, required=False, default="/mnt/18T/liwx/H800-Mount/DataSet/DataSet_Smoke_V2/smoke_all/images/val", help='测试路径')
    parser.add_argument('-m', '--model', type=str, choices=['dfine', 'deim', 'yolo'], default="dfine", help='选择加载的模型: dfine/deim/yolo, gt模式的label_mapping与此有关')
    parser.add_argument('-mo', '--mode',type=str, choices=['xml', 'gt', 'pred'], default='gt', help='选择模式: xml/gt/pred')
    parser.add_argument('-mp', '--model-path', type=str, required=False, default="", help='模型加载路径')
    # generate xml from model
    parser.add_argument('-th', '--nms-thres', type=float, default=0.7, help='nms阈值: float/None, 分别代表固定阈值和默认的置信度字典') 
    parser.add_argument('--nms-class', default="all", help='nms过滤的类别: # None/all/[1,2,3], 分别代表不过滤/全过滤/过滤指定类别') 
    parser.add_argument('--xml-output', type=str, default=None, help='xml结果保存位置, 默认为 ./datasets/{project_name}/xmls')

    # generate gt from xml
    parser.add_argument('--gt-output', type=str, default=None, help='gt结果保存位置, 默认为 ./datasets/{project_name}/gt')

    # generate pred from yolo
    parser.add_argument('--pred-output', type=str, default=None, help='gt结果保存位置, 默认为 ./datasets/{project_name}/pred')
    parser.add_argument('--conf', type=float, default=0.4, help='yolo预测置信度阈值, 默认0.35')
    parser.add_argument('--img_size', type=float, default=-1, help='yolo预测置信度阈值, 默认640')

    return parser.parse_args()

'''
1. 先使用模型生成预测结果，并以txt保存
2. 读取配置文件中xml,生成gt,并以txt文件形式保存
3. 执行下一个脚本
'''
def main():
    args = parse_args()
    print(args)
    if args.mode == "xml" or (args.mode == "pred" and (args.model == "dfine" or args.model == "deim")):
        generate_xml(args)          # python3 generate.py -mo pred -m deim -c projects/smoke.yaml  生成预测结果以txt形式保存
    elif args.mode == "xml" and args.model == "yolo":
        convert_yolo_2_xml(args)    # yolo格式（txt文件） 转换成 pascalvoc格式（xml） （第一步）,如果有直接执行第二步
    elif args.mode == "gt":
        generate_gt(args)           # 根据 标注的 xml 生成 gt (classname 左上角坐标 右下角坐标) 以txt文件形式保存在当前目录datasets下  （第二步） 
        #python3 generate.py -mo gt -c /mnt/18T/liwx/H800-Mount/Code/Experiment/tools/detection_metrics/projects/smoke_yolo.yaml -m yolo
    elif args.mode == "pred" and args.model == "yolo":
        generate_pred(args)         #  第三步 ，生成预测文件 classname 左上角坐标 右下角坐标
 
if __name__ == '__main__':
    main()
