import os
import cv2
import argparse
import sys
import yaml
from tqdm import tqdm
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

def create_voc_xml(img_path, txt_path, class_mapping, output_xml_path):
    """
    为单个图片和其YOLO标注创建PASCAL VOC XML文件。
    """
    # 1. 读取图片信息
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}。跳过此文件。")
            return
        img_h, img_w, img_d = img.shape
    except Exception as e:
        print(f"错误: 读取图片 {img_path} 时出错: {e}。跳过此文件。")
        return

    # 2. 创建XML根节点和基本信息
    node_root = Element('annotation')
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = os.path.basename(os.path.dirname(img_path))
    
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(img_path)
    
    node_path = SubElement(node_root, 'path')
    node_path.text = img_path
    
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Unknown'
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img_w)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img_h)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(img_d)
    
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    # 3. 读取YOLO标注文件并转换
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"警告: 在 {txt_path} 中发现格式不正确的行: '{line.strip()}'。已忽略。")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    print(f"警告: 在 {txt_path} 中解析行内容失败: '{line.strip()}'。已忽略。")
                    continue

                # 检查class_id是否存在于映射中
                if class_id not in class_mapping:
                    print(f"警告: 在 {txt_path} 中发现无效的类别ID: {class_id} (未在config.yaml中定义)。已忽略。")
                    continue
                
                class_name = class_mapping[class_id]
                
                # YOLO坐标转换 -> VOC坐标 (xmin, ymin, xmax, ymax)
                abs_x_center = x_center * img_w
                abs_y_center = y_center * img_h
                abs_width = width * img_w
                abs_height = height * img_h
                
                xmin = int(abs_x_center - abs_width / 2)
                ymin = int(abs_y_center - abs_height / 2)
                xmax = int(abs_x_center + abs_width / 2)
                ymax = int(abs_y_center + abs_height / 2)

                # 坐标边界检查，确保不超出图片范围
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)
                
                # 创建object节点
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = class_name
                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'
                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(xmin)
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(ymin)
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(xmax)
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(ymax)

    # 4. 格式化并写入XML文件
    xml_string = tostring(node_root, 'utf-8')
    dom = parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent='    ')
    
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def convert_yolo_2_xml(args):
    """
    主执行函数，负责读取配置、遍历文件并调用转换函数。
    """
    # 1. 读取配置文件
    config_path = args.config_path
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 '{config_path}' 不存在。")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"错误: 解析配置文件 '{config_path}'失败: {e}")
            sys.exit(1)

    # 2. 从配置中获取路径和映射
    try:
        image_folder = config['paths']['image_folder']
        label_folder = config['paths']['label_folder']
        xml_save_path = config['xml_save_path']
        label_mapping = config['yolo_mapping']
    except KeyError as e:
        print(f"错误: 配置文件 '{config_path}' 中缺少必要的键: {e}")
        sys.exit(1)
        
    # 3. 校验输入路径
    if not os.path.isdir(image_folder):
        print(f"错误: 图片目录 '{image_folder}' 不存在或不是一个目录。")
        sys.exit(1)
    if not os.path.isdir(label_folder):
        print(f"错误: 标签目录 '{label_folder}' 不存在或不是一个目录。")
        sys.exit(1)
        
    # 4. 创建输出目录
    os.makedirs(xml_save_path, exist_ok=True)
    print(f"配置文件加载成功，XML文件将保存到: {xml_save_path}")

    # 5. 开始转换
    txt_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"警告: 在目录 '{label_folder}' 中未找到任何 .txt 标注文件。")
        return

    for txt_file in tqdm(txt_files, desc="转换进度"):
        base_filename = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(label_folder, txt_file)
        
        # 查找对应的图片文件 (支持多种常见格式)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            potential_img_path = os.path.join(image_folder, base_filename + ext)
            if os.path.exists(potential_img_path):
                img_path = potential_img_path
                break
        
        if img_path is None:
            print(f"\n警告: 找不到与 {txt_file} 对应的图片文件。跳过此文件。")
            continue
            
        # 定义输出XML文件路径
        output_xml_path = os.path.join(xml_save_path, base_filename + '.xml')
        
        # 执行转换
        create_voc_xml(img_path, txt_path, label_mapping, output_xml_path)

    print(f"\n转换完成！总共处理了 {len(txt_files)} 个标注文件。")


