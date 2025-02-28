import os, shutil
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
label_a = {
    0: "fire",
    1: "smoke",
    2: "nest",
    3: "dxyw",
    4: "coated_steel_sheet",
    5: "dustproof_net",
    6: "tower_crane",
    7: "truck_crane",
    8: "bulldozer",
    9: "pump_truck",
    10: "cement_mixer_truck",
    11: "truck",
    12: "excavator",
    13: "greenhouse",
    14: "tarpaulin",
    15: "piling_rig",
    16: "oil_tanker_truck",
    17: "truck_crane_b"
    }
label_b = {value: key for key, value in label_a.items()}

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

# 定义输入和输出路径
image_folder = r"/data1/datasets/shudian_20000/images/test/"  # 包含所有图片名的文本文件
xml_folder = r"/data1/datasets/shudian_20000/annotations/test_coated_dustproof/"              # XML 文件夹
gt_folder = r'/data1/home/fz/desktop/detection_metrics_new/txts/shudian_20000/gt/'                # 输出文件夹

if os.path.exists(gt_folder):
    shutil.rmtree(gt_folder)
os.makedirs(gt_folder, exist_ok=True)
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
# print(len(image_paths), len(xml_paths))
# assert len(image_paths) == len(xml_paths), "标签和图片数量不等"  

ori_classes = {}
unuse_classes = {}
new_classes = {}
all_numbers = 0
unuse_numbers = 0
error_numbers = 0
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
                if class_name in ["coated_steel_sheet_a"]:
                    class_name = "coated_steel_sheet" 
                elif "fandouche" == class_name:
                    class_name = "truck" 
                elif "roller" == class_name:
                    class_name = "bulldozer" 
                elif "long_excavator" == class_name:
                    class_name = "excavator"
                elif class_name in ["sgjx", "ta", "dxdg", "sgqx", "bridge_crane", "xxyw", "fnss", "construction_machinery", "fzc", "other", "xxdy"]:
                    unuse_numbers += 1
                    if class_name not in unuse_classes:
                        unuse_classes[class_name] = 1
                    else:
                        unuse_classes[class_name] += 1
                    continue
                else:
                    class_name = class_name           
                bndbox = obj.find('bndbox')
                box = (int(float(bndbox.find('xmin').text)), int(float(bndbox.find('ymin').text)), int(float(bndbox.find('xmax').text)),
                    int(float(bndbox.find('ymax').text)))
                x1 = min(max(box[0], 0), w)  # 物体中心点x的坐标比(相当于 x/原图w)
                y1 = min(max(box[1], 0), h)  # 物体宽度的宽度比(相当于 w/原图w)
                x2 = min(max(box[2], 0), w)  # 物体中心点y的坐标比(相当于 y/原图h)
                y2 = min(max(box[3], 0), h)  # 物体宽度的宽度比(相当于 h/原图h)
                out_f.write(class_name + " " + " ".join([str(a) for a in [x1,y1,x2,y2]]) + '\n')
                if class_name not in new_classes:
                    new_classes[class_name] = 1
                else:
                    new_classes[class_name] += 1 
print(all_numbers)
print(unuse_numbers)
print(error_numbers)
print(ori_classes, len(ori_classes.keys()), sum(ori_classes.values()))
print(unuse_classes, len(unuse_classes.keys()), sum(unuse_classes.values()))
print(new_classes, len(new_classes.keys()), sum(new_classes.values()))
line_count = 0
txt_count = 0
for txt_root, txt_dirs, txt_files in os.walk(gt_folder):
    for txt_file in txt_files:
        txt_path = os.path.join(txt_root, txt_file)
        txt_count += 1
        with open(txt_path, 'r', encoding='utf-8') as file:
            line_count += len(file.readlines())
print(line_count, sum(new_classes.values()), txt_count, len(image_paths), line_count + unuse_numbers, sum(ori_classes.values()))
assert txt_count == len(image_paths) and line_count == sum(new_classes.values()) and (line_count + unuse_numbers) == sum(ori_classes.values()), "核对标签数量"
    