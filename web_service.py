from flask import Flask, render_template, send_file, request, redirect, url_for, jsonify
import os
import json
import argparse
import cv2
import xml.etree.ElementTree as ET
import logging
import subprocess

app = Flask(__name__)

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 使用 argparse 从命令行获取 JSON 文件目录
def parse_args():
    parser = argparse.ArgumentParser(description='启动 Web 服务以展示错误分析结果')
    parser.add_argument('-j', '--json-dir', type=str, required=True, help='结构化数据的 JSON 文件目录')
    return parser.parse_args()

# 获取命令行参数
args = parse_args()
JSON_DIR = args.json_dir
MODIFIED_XML_DIR = os.path.join(JSON_DIR, 'modified_xmls')
os.makedirs(MODIFIED_XML_DIR, exist_ok=True)

@app.route('/')
def index():
    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    return render_template('index.html', json_files=json_files)

@app.route('/view/<json_file>/<int:index>')
def view(json_file, index):
    # 加载指定的 JSON 文件
    json_path = os.path.join(JSON_DIR, json_file)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保索引在范围内
    if index < 0:
        index = len(data) - 1
    elif index >= len(data):
        index = 0
    
    item = data[index]
    return render_template('view.html', item=item, index=index, json_file=json_file, total=len(data), data=data)

@app.route('/image/<image_name>')
def image(image_name):
    # 从 JSON 数据中提取图片路径
    for json_file in os.listdir(JSON_DIR):
        json_path = os.path.join(JSON_DIR, json_file)
        # 确保 json_path 是一个文件路径
        if not os.path.isfile(json_path):
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item['image_name'] == image_name:
                    image_path = item['image_path']
                    if os.path.exists(image_path):
                        # 读取图片
                        img = cv2.imread(image_path)
                        # 绘制 FP 框
                        for idx, fp in enumerate(item['fp']):
                            x1, y1, x2, y2 = map(int, fp['bbox'])
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            # 计算文字背景
                            text = f"FP-{idx+1}"
                            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            # 确保文字框在图片范围内
                            text_y = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5
                            cv2.rectangle(img, (x1, text_y - text_height), (x1 + text_width, text_y), (0, 0, 255), -1)
                            cv2.putText(img, text, (x1, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # 绘制 FN 框
                        for idx, fn in enumerate(item['fn']):
                            x1, y1, x2, y2 = map(int, fn['bbox'])
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            # 计算文字背景
                            text = f"FN-{idx+1}"
                            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            # 确保文字框在图片范围内
                            text_y = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5
                            cv2.rectangle(img, (x1, text_y - text_height), (x1 + text_width, text_y), (255, 0, 0), -1)
                            cv2.putText(img, text, (x1, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # 保存临时图片
                        temp_image_path = f"/tmp/{image_name}.jpg"
                        cv2.imwrite(temp_image_path, img)
                        return send_file(temp_image_path)
    return "Image not found", 404

@app.route('/save/<json_file>/<int:index>', methods=['POST'])
def save(json_file, index):
    # 获取用户选择的数据
    user_data = request.json
    json_path = os.path.join(JSON_DIR, json_file)
    
    # 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 更新数据
    for i, fp in enumerate(data[index]['fp']):
        # 确保 user_data['fp'][i] 是布尔值
        fp['save'] = bool(user_data['fp'][i])
    for i, fn in enumerate(data[index]['fn']):
        # 确保 user_data['fn'][i] 是布尔值
        fn['save'] = bool(user_data['fn'][i])
    
    # 保存更新后的数据
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return jsonify({"status": "success"})

@app.route('/modify_xmls', methods=['POST'])
def modify_xmls():
    # 获取请求中的JSON文件名
    json_file = request.json.get('json_file')
    if not json_file:
        return jsonify({"status": "error", "message": "No JSON file specified"}), 400
    
    json_path = os.path.join(JSON_DIR, json_file)
    if not os.path.exists(json_path):
        return jsonify({"status": "error", "message": "JSON file not found"}), 404
    
    # 调用 modify_xmls.py 脚本
    subprocess.run(['python', 'modify_xmls.py', '--json', json_path, '--xml-dir', MODIFIED_XML_DIR])
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=10086) 