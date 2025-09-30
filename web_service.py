from flask import Flask, render_template, send_file, request, redirect, url_for, jsonify
import os
import json
import argparse
import cv2
import xml.etree.ElementTree as ET
import logging
import subprocess
import sys  # <<< FIX: Import sys to get the current python executable
from io import BytesIO # <<< OPTIMIZATION: Import BytesIO for in-memory image handling
import uuid # <<< FIX: Import uuid for unique temporary filenames

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
    json_path = os.path.join(JSON_DIR, json_file)
    if not os.path.exists(json_path):
        # 如果文件不存在，直接渲染错误页面
        return render_template('error.html', message=f"错误: JSON文件 '{json_file}' 未找到。"), 404

    # <<< FIX: Add error handling for corrupted JSON files
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # 如果文件为空，json.load 会直接报错
            data = json.load(f)
        if not data: # 检查JSON文件是否是空的列表或字典
             return render_template('error.html', message=f"错误: JSON文件 '{json_file}' 内容为空。")
    except json.JSONDecodeError:
        logging.error(f"JSON文件解析失败: {json_path}。文件可能已损坏或为空。")
        return render_template('error.html', message=f"错误: JSON文件 '{json_file}' 已损坏或格式不正确。")
    except Exception as e:
        logging.error(f"加载 {json_path} 时发生未知错误: {e}")
        return render_template('error.html', message=f"加载文件时发生未知错误: {e}")

    # 确保索引在范围内
    if index < 0:
        index = len(data) - 1
    elif index >= len(data):
        index = 0
    
    item = data[index]
    return render_template('view.html', item=item, index=index, json_file=json_file, total=len(data), data=data)

@app.route('/image/<image_name>')
def image(image_name):
    # <<< OPTIMIZATION: Generate image in memory instead of writing to disk
    json_file = request.args.get('json_file')
    if not json_file:
        return "JSON file not specified", 400
        
    json_path = os.path.join(JSON_DIR, json_file)
    if not os.path.exists(json_path):
        return "JSON file not found", 404
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return "Failed to load or find JSON data", 404

    for item in data:
        if item['image_name'] == image_name:
            image_path = item['image_path']
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                # 绘制 FP 框
                for idx, fp in enumerate(item.get('fp', [])): # Use .get for safety
                    x1, y1, x2, y2 = map(int, fp['bbox'])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f"FP-{idx+1}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_y = y1 - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5
                    cv2.rectangle(img, (x1, text_y - text_height), (x1 + text_width, text_y + 2), (0, 0, 255), -1)
                    cv2.putText(img, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # 绘制 FN 框
                for idx, fn in enumerate(item.get('fn', [])): # Use .get for safety
                    x1, y1, x2, y2 = map(int, fn['bbox'])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    text = f"FN-{idx+1}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_y = y1 - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5
                    cv2.rectangle(img, (x1, text_y - text_height), (x1 + text_width, text_y + 2), (255, 0, 0), -1)
                    cv2.putText(img, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 将图片编码到内存缓冲区
                is_success, buffer = cv2.imencode(".jpg", img)
                if is_success:
                    # 从内存直接发送文件
                    return send_file(BytesIO(buffer), mimetype='image/jpeg')
                else:
                    return "Failed to encode image", 500
    return "Image data not found in JSON", 404

@app.route('/save/<json_file>/<int:index>', methods=['POST'])
def save(json_file, index):
    
    #在 Python 执行完 open() 之后，操作系统已经根据 'w' 模式的指令，
    #    将 json_path 指向的那个文件内容清空了（文件大小变为 0 字节）。
    #    此时，文件已经处于一个非常脆弱的“空”状态。
    #
    #    <<<<<<<<<<<<<<<<<<<<<<<<<<< 危险区开始 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #
    #    如果在这里发生以下任何一种情况：
    #    - 进程被强制杀死 (kill -9)
    #    - 服务器突然断电
    #    - 操作系统崩溃
    #    - Python 解释器自身出现致命错误而退出
    #
    #    那么，这个空的 json 文件就会被永久地保留下来。
    #    当下一次有请求访问这个文件时，json.load() 就会因为读到空内容而报错。
    #
    #    <<<<<<<<<<<<<<<<<<<<<<<<<<< 危险区结束 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # 3. 如果程序幸运地没有在“危险区”崩溃，它会继续执行这一行。
    #    这一步会把内存中的 `data` 对象序列化成字符串，并写入到那个已经被清空的文件中。
    # <<< FIX: Implement atomic write to prevent file corruption
    try:
        user_data = request.json
        if not user_data:
            return jsonify({"status": "error", "message": "No data received"}), 400
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON data in request"}), 400

    json_path = os.path.join(JSON_DIR, json_file)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果原始文件损坏或不存在，则无法保存
        logging.error(f"无法加载原始JSON文件进行保存: {json_path}")
        return jsonify({"status": "error", "message": "Original JSON file is missing or corrupted."}), 500

    # 更新数据
    try:
        for i, fp in enumerate(data[index]['fp']):
            fp['save'] = bool(user_data['fp'][i])
        for i, fn in enumerate(data[index]['fn']):
            fn['save'] = bool(user_data['fn'][i])
    except (IndexError, KeyError) as e:
        logging.error(f"保存时数据结构不匹配: {e}")
        return jsonify({"status": "error", "message": "Data structure mismatch."}), 400

    # 原子写入过程
    temp_file_path = f"{json_path}.{uuid.uuid4()}.tmp"
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        # 写入成功后，用临时文件替换原始文件
        os.replace(temp_file_path, json_path)
    except Exception as e:
        logging.error(f"保存文件失败: {e}")
        # 如果发生错误，尝试删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify({"status": "error", "message": "Failed to save file."}), 500
    
    logging.info(f"成功更新并保存文件: {json_path}")
    return jsonify({"status": "success"})


@app.route('/modify_xmls', methods=['POST'])
def modify_xmls():
    json_file = request.json.get('json_file')
    if not json_file:
        return jsonify({"status": "error", "message": "No JSON file specified"}), 400
    
    json_path = os.path.join(JSON_DIR, json_file)
    if not os.path.exists(json_path):
        return jsonify({"status": "error", "message": "JSON file not found"}), 404
    
    # <<< OPTIMIZATION: Use sys.executable to ensure correct python interpreter
    try:
        # 使用 sys.executable 保证调用的是当前虚拟环境的 python
        subprocess.run(
            [sys.executable, 'modify_xmls.py', '--json', json_path, '--xml-dir', MODIFIED_XML_DIR],
            check=True, # 如果脚本返回非零退出码，则会引发异常
            capture_output=True, text=True # 捕获输出，便于调试
        )
        logging.info(f"modify_xmls.py 脚本成功执行: {json_file}")
        return jsonify({"status": "success"})
    except subprocess.CalledProcessError as e:
        logging.error(f"modify_xmls.py 脚本执行失败: {e}")
        logging.error(f"脚本标准输出: {e.stdout}")
        logging.error(f"脚本标准错误: {e.stderr}")
        return jsonify({"status": "error", "message": f"Script failed: {e.stderr}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=10086, host='0.0.0.0')