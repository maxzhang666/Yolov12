# YOLO12 目标检测 - 完整使用指南

## 🎯 项目概述

这是一个基于 YOLOv12 的人物和头部检测项目，专门针对 MacBook Air M1 优化。

**检测目标**: 
- 👤 person（完整人物）
- 👨 head（人物头部）

**数据集**:
- 格式: YOLO
- 类别数: 2
- 已划分: 训练集/验证集/测试集

---

## 📦 安装步骤

### 方式一: 自动安装（推荐）

```bash
# 运行自动安装脚本
./setup.sh

# 激活虚拟环境
source venv/bin/activate
```

### 方式二: 手动安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 🚀 训练模型

### 基础训练（推荐）

```bash
# 激活虚拟环境
source venv/bin/activate

# 开始训练
python train_yolo.py
```

训练会自动：
- ✅ 下载预训练权重（首次运行）
- ✅ 使用 MPS 加速（Apple Silicon）
- ✅ 自动保存最佳模型
- ✅ 生成训练曲线图
- ✅ 在验证集上评估

### 预期训练时间

| Epochs | 预计时间 | 适用场景 |
|--------|---------|---------|
| 5      | 5-10分钟 | 快速测试 |
| 50     | 30-60分钟 | 标准训练（推荐）|
| 100    | 1-2小时 | 高质量训练 |

### 自定义训练参数

编辑 `train_yolo.py` 文件：

```python
# 找到配置部分（约第15行）
epochs = 50              # 改为你想要的轮数
batch_size = 16          # 如果内存不足，改为8
img_size = 640           # 图像尺寸
```

---

## 📊 监控训练

### 方式一: 实时监控

在训练过程中，打开新终端窗口：

```bash
source venv/bin/activate
python monitor.py watch
```

每5秒刷新一次训练进度。

### 方式二: 查看摘要

```bash
python monitor.py summary
```

显示训练结果摘要。

### 方式三: 绘制曲线

```bash
python monitor.py plot
```

生成自定义训练曲线图。

---

## 🔍 测试模型

训练完成后，测试模型性能：

```bash
python test_yolo.py
```

这会：
1. ✅ 在测试集上评估模型
2. ✅ 生成预测结果图片
3. ✅ 输出性能指标（mAP、精确率、召回率）

---

## 📈 查看结果

训练结果保存在 `runs/detect/yolo12n_person_head/` 目录：

```
runs/detect/yolo12n_person_head/
├── weights/
│   ├── best.pt          ⭐ 最佳模型（用于部署）
│   └── last.pt          最后一次训练的权重
├── results.png          📊 训练曲线
├── confusion_matrix.png 🎯 混淆矩阵
├── PR_curve.png         📈 精度-召回率曲线
├── F1_curve.png         📈 F1分数曲线
└── results.csv          📄 详细训练数据
```

### 重要文件说明

| 文件 | 用途 |
|------|------|
| **best.pt** | 验证集上表现最好的模型，用于实际部署 |
| **last.pt** | 最后一个epoch的模型，用于继续训练 |
| **results.png** | 查看训练是否正常，有无过拟合 |
| **confusion_matrix.png** | 查看哪些类别容易混淆 |

---

## 💡 使用模型进行预测

### 1. 单张图片预测

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/yolo12n_person_head/weights/best.pt')

# 预测单张图片
results = model.predict('path/to/image.jpg', save=True, conf=0.25)

# 查看结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"类别: {result.names[int(box.cls)]}, 置信度: {box.conf[0]:.2%}")
```

### 2. 批量预测

```python
# 预测整个文件夹
results = model.predict('path/to/images/', save=True, conf=0.25)
```

### 3. 视频预测

```python
# 预测视频
results = model.predict('path/to/video.mp4', save=True, conf=0.25)
```

### 4. 摄像头实时检测

```python
# 使用摄像头
results = model.predict(source=0, show=True, conf=0.25)
```

---

## 🔧 常见问题

### Q1: 训练时提示 "MPS不可用"

**解决方案**:
```python
# 编辑 train_yolo.py，找到 device 设置
device = 'cpu'  # 改为使用CPU
```

虽然CPU较慢，但依然可以训练。

---

### Q2: 内存不足 / Out of Memory

**解决方案**:
```python
# 在 train_yolo.py 中降低参数
batch_size = 8    # 从16降到8
workers = 2       # 从4降到2
```

---

### Q3: 训练速度太慢

**优化建议**:
1. 确认MPS已启用（看训练开始时的输出）
2. 减少数据增强
3. 使用较小的图像尺寸（320）
4. 考虑使用云端GPU（Google Colab免费）

---

### Q4: 效果不理想

**改进方向**:

**短期（调参）**:
```python
epochs = 100           # 增加训练轮数
```

**中期（模型）**:
```python
model_name = 'yolo12s.pt'  # 使用更大的模型
```

**长期（数据）**:
- 增加训练数据量
- 改进数据质量
- 增强数据多样性

---

### Q5: 如何继续训练

```python
# 从上次停止的地方继续
model = YOLO('runs/detect/yolo12n_person_head/weights/last.pt')
model.train(data='datasets/data.yaml', epochs=50, resume=True)
```

---

## 🎓 性能评估指标说明

### mAP50-95（最重要）
- 综合评估指标
- 值越高越好（0-1之间）
- **0.5-0.6**: 良好
- **0.6-0.7**: 优秀
- **>0.7**: 非常优秀

### mAP50
- 较宽松的评估标准
- 通常比 mAP50-95 高

### Precision（精确率）
- 预测为正的样本中，真正为正的比例
- 高精确率 = 误报少

### Recall（召回率）
- 所有正样本中，被正确预测的比例
- 高召回率 = 漏检少

---

## 🚀 下一步

### 如果效果满意（mAP50-95 > 0.5）

1. **直接部署使用**
   ```python
   model = YOLO('runs/detect/yolo12n_person_head/weights/best.pt')
   ```

2. **导出为其他格式**
   ```python
   # 导出为ONNX（跨平台）
   model.export(format='onnx')
   
   # 导出为CoreML（iOS/macOS）
   model.export(format='coreml')
   ```

3. **优化推理速度**
   - 使用INT8量化
   - 导出为TensorRT（NVIDIA GPU）

### 如果效果不满意

1. **增加训练轮数**
   ```python
   epochs = 100
   ```

2. **使用更大模型**
   ```python
   model_name = 'yolo12s.pt'  # 或 yolo12m.pt
   ```

3. **收集更多数据**
   - 增加训练样本
   - 增加场景多样性

### 如果需要更快训练

1. **使用云端GPU**
   - Google Colab（免费T4 GPU）
   - Kaggle（免费P100 GPU）
   - AWS/阿里云（付费）

2. **修改训练脚本的device**
   ```python
   device = 'cuda'  # 使用NVIDIA GPU
   ```

---

## 📚 学习资源

- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [YOLO 训练技巧](https://docs.ultralytics.com/guides/training-tips/)
- [模型导出指南](https://docs.ultralytics.com/modes/export/)
- [数据集准备指南](https://docs.ultralytics.com/datasets/)

---

## 📞 获取帮助

遇到问题？

1. 查看本文档的"常见问题"部分
2. 检查训练日志中的错误信息
3. 查看 Ultralytics 官方文档
4. 在 GitHub Issues 提问

---

## ✅ 快速检查清单

在开始训练前，确认：

- [ ] Python 3.8+ 已安装
- [ ] 虚拟环境已创建并激活
- [ ] 所有依赖已安装（`pip list`）
- [ ] 数据集在 `datasets/` 目录下
- [ ] `data.yaml` 配置正确
- [ ] MPS 可用（macOS 12.3+，PyTorch 2.0+）

开始训练：

- [ ] 运行 `python train_yolo.py`
- [ ] 监控训练进度
- [ ] 等待训练完成（30-60分钟）

训练完成后：

- [ ] 查看 `results.png` 训练曲线
- [ ] 运行 `python test_yolo.py` 评估
- [ ] 检查 mAP50-95 指标
- [ ] 使用 `best.pt` 进行预测

---

**祝您训练顺利！如有问题随时查阅本指南。** 🎉
