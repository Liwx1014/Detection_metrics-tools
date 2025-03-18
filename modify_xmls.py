'''
读取--json指定的json文件，读取xml文件，根据json文件中的fp和fn，修改xml文件中的目标框，保存修改后的xml文件到--xml-dir指定的目录中
其中每个json文件中包含一个image_name，image_path，fp，fn，original_xml_path，class_name，示例如下
{
        "image_name": "shudian_00002901",
        "image_path": "/data1/datasets/shudian_2025/images/test/shudian_00002901.jpg",
        "fp": [
            {
                "bbox": [
                    966.0,
                    0.0,
                    1066.0,
                    138.0
                ],
                "confidence": 0.591,
                "save": false
            }
        ],
        "fn": [
            {
                "bbox": [
                    730.0,
                    401.0,
                    810.0,
                    513.0
                ],
                "save": false
            }
        ],
        "original_xml_path": "/data1/datasets/shudian_2025/annotations/test/shudian_00002901.xml",
        "class_name": "dxyw"
    },

'''

import json
import os
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Modify XML files based on JSON input")
    parser.add_argument('-j', '--json', required=True, help='Path to the JSON file')
    parser.add_argument('-x', '--xml-dir', required=True, help='Directory to save modified XML files')
    return parser.parse_args()

def modify_xml(json_data, xml_dir):
    for item in tqdm(json_data):
        image_name = item['image_name']
        original_xml_path = item['original_xml_path']
        fp = item['fp']
        fn = item['fn']
        class_name = item['class_name']

        # Load the original XML
        tree = ET.parse(original_xml_path)
        root = tree.getroot()

        # 标记是否需要保存修改后的XML
        need_save = False

        # Modify the XML based on fp and fn
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == class_name:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # Check for false negatives
                for fn_item in fn:
                    fn_bbox = fn_item['bbox']
                    if not fn_item['save']:
                        if (xmin == fn_bbox[0] and ymin == fn_bbox[1] and
                            xmax == fn_bbox[2] and ymax == fn_bbox[3]):
                            root.remove(obj)
                            need_save = True

        # Add false positives
        for fp_item in fp:
            if fp_item['save']:
                fp_bbox = fp_item['bbox']
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = class_name
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(fp_bbox[0])
                ET.SubElement(bndbox, 'ymin').text = str(fp_bbox[1])
                ET.SubElement(bndbox, 'xmax').text = str(fp_bbox[2])
                ET.SubElement(bndbox, 'ymax').text = str(fp_bbox[3])
                need_save = True

        # 只有在需要保存时才写入新的XML文件
        if need_save:
            modified_xml_path = os.path.join(xml_dir, f"{image_name}.xml")
            # 确保目录存在
            os.makedirs(os.path.dirname(modified_xml_path), exist_ok=True)
            tree.write(modified_xml_path, encoding='utf-8', xml_declaration=True)

def main():
    args = parse_args()
    with open(args.json, 'r') as f:
        json_data = json.load(f)
    os.makedirs(args.xml_dir, exist_ok=True)
    modify_xml(json_data, args.xml_dir)

if __name__ == "__main__":
    main()