import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import yaml
import argparse
import sys

def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度
    h = box[3] - box[2]  # 物体实际像素高度
    x = min(max(x * dw, 0), 1)  # 物体中心点x的坐标比(相当于 x/原图w)
    w = min(max(w * dw, 0), 1)  # 物体宽度的宽度比(相当于 w/原图w)
    y = min(max(y * dh, 0), 1)  # 物体中心点y的坐标比(相当于 y/原图h)
    h = min(max(h * dh, 0), 1)  # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]

def generate_gt(args):
    # 读取配置文件
    config_path = args.config_path
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取项目名称，如果未配置则使用default
    project_name = config.get('project_name', 'default')
    
    # 获取配置信息
    if args.model == "yolo":
        label_mapping = config['yolo_mapping']
    else:
        label_mapping = config['transform_mapping']
    image_folder = config['paths']['image_folder']
    xml_folder = config['paths']['xml_folder']
    gt_folder = args.gt_output if args.gt_output else os.path.join('./datasets', project_name, 'gt')
    class_mapping = config['class_mapping']
    
    # 获取标签尺寸过滤阈值
    filter_config = config.get('filter_size', {})
    min_size = filter_config.get('min_size', 0)  # 默认为0表示不过滤
    min_edge = filter_config.get('min_edge', 0)  # 默认为0表示不过滤
    
    # 获取所有有效的类别（label_mapping中定义的类别）
    if 'valid_list' in config and config['valid_list']:
        valid_classes = config['valid_list']
    else:
        valid_classes = set(label_mapping.values())

    if os.path.exists(gt_folder):
        shutil.rmtree(gt_folder)
    os.makedirs(gt_folder, exist_ok=True)

    # 获取所有图片和XML文件路径
    image_paths = []
    xml_paths = []
    for img_root, img_dirs, img_files in os.walk(image_folder):
        for img_file in img_files:
            image_path = os.path.join(img_root, img_file)
            image_paths.append(image_path)
    for xml_root, xml_dirs, xml_files in os.walk(xml_folder):
        for xml_file in xml_files:
            xml_path = os.path.join(xml_root, xml_file)
            xml_paths.append(xml_path)

    ori_classes = {}
    unuse_classes = {}
    new_classes = {}
    all_numbers = 0
    unuse_numbers = 0
    error_numbers = 0
    filtered_count = 0  # 记录被过滤的标签数量

    for image_path in tqdm(image_paths, desc="Processing images"):
        w, h = Image.open(image_path).size
        dir_path, image_name = os.path.split(image_path)
        image_name = os.path.splitext(image_name)[0]
        xml_path = os.path.join(xml_folder, image_name + '.xml')
        
        if os.path.isfile(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                if width != w or height != h:
                    print(image_path, xml_path, w, h, width, height)
                    error_numbers += 1
                    
            with open(os.path.join(gt_folder, image_name + '.txt'), 'w', encoding='utf-8') as out_f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    all_numbers += 1
                    
                    if class_name not in ori_classes:
                        ori_classes[class_name] = 1
                    else:
                        ori_classes[class_name] += 1
                        
                    # 处理类别映射
                    if class_name in class_mapping:
                        new_class_name = class_mapping[class_name]
                    else:
                        new_class_name = class_name
                    # 如果类别不在有效类别列表中，则忽略
                    if new_class_name not in valid_classes:
                        unuse_numbers += 1
                        if class_name not in unuse_classes:
                            unuse_classes[class_name] = 1
                        else:
                            unuse_classes[class_name] += 1
                        continue
                        
                    bndbox = obj.find('bndbox')
                    box = (
                        int(float(bndbox.find('xmin').text)),
                        int(float(bndbox.find('ymin').text)),
                        int(float(bndbox.find('xmax').text)),
                        int(float(bndbox.find('ymax').text))
                    )
                    
                    # 计算边界框的宽度和高度
                    box_w = box[2] - box[0]
                    box_h = box[3] - box[1]
                    
                    # 根据尺寸阈值过滤标签
                    if (min_size > 0 and box_w < min_size and box_h < min_size) or \
                       (min_edge > 0 and (box_w < min_edge or box_h < min_edge)):
                        filtered_count += 1
                        continue
                    
                    x1 = min(max(box[0], 0), w)
                    y1 = min(max(box[1], 0), h)
                    x2 = min(max(box[2], 0), w)
                    y2 = min(max(box[3], 0), h)
                    
                    out_f.write(f"{new_class_name} {x1} {y1} {x2} {y2}\n")
                    if new_class_name is not None:
                        if new_class_name not in new_classes:
                            new_classes[new_class_name] = 1
                        else:
                            new_classes[new_class_name] += 1

    # 打印统计信息
    print(f"总标注数量: {all_numbers}")
    print(f"未使用的标注数量: {unuse_numbers}")
    print(f"被过滤的标注数量: {filtered_count}")  # 添加过滤数量统计
    print(f"图片尺寸不匹配数量: {error_numbers}")
    print(f"原始类别统计: {ori_classes}, 类别数: {len(ori_classes)}, 总数: {sum(ori_classes.values())}")
    print(f"未使用类别统计: {unuse_classes}, 类别数: {len(unuse_classes)}, 总数: {sum(unuse_classes.values())}")
    print(f"新类别统计: {new_classes}, 类别数: {len(new_classes)}, 总数: {sum(new_classes.values())}")

    # 验证结果
    line_count = 0
    txt_count = 0
    for txt_root, txt_dirs, txt_files in os.walk(gt_folder):
        for txt_file in txt_files:
            txt_path = os.path.join(txt_root, txt_file)
            txt_count += 1
            with open(txt_path, 'r', encoding='utf-8') as file:
                line_count += len(file.readlines())
                
    print(f"txt文件行数: {line_count}, 新类别总数: {sum(new_classes.values())}, txt文件数: {txt_count}, "
          f"图片数: {len(image_paths)}, 总标注数: {line_count + unuse_numbers}, 原始标注总数: {sum(ori_classes.values())}")
    

if __name__ == "__main__":
    main() 