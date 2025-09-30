# detection_metrics
用于测试目标检测任务mAP

## 使用方法
### 第一步 generate.py
生成预测结果，以txt形式保存（不是归一化格式），模型名称可以指定
1. python3 generate.py -mo pred -m deim -c projects/smoke.yaml  
2. 如果是yolo模型：需要先转成xml,在指定第一步
3. 根据 标注的 xml 生成 gt (classname 左上角坐标 右下角坐标) 以txt文件形式保存在当前目录datasets下  
python3 generate.py -mo gt -c /mnt/18T/liwx/H800-Mount/Code/Experiment/tools/detection_metrics/projects/smoke_yolo.yaml -m yolo
4. 生成预测结果txt

### 第二步 pascalvoc.py 计算指标
1.  python pascalvoc.py -g datasets/yiwu_6cls/gt -p datasets/yiwu_6cls/deim_pred  -c projects/yiwu_6cls.yaml --deep-analysis tower_crane'    
上述命令会比对预测结果和真值，并生成详细指标信息，并生成具体FP和FN的结构文件

### 第三步 web_service.py 分析误报和漏报







