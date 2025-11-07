# YOLO12 Person & Head Detection

基于 YOLOv12 的人物和头部检测项目，针对 MacBook Air M1 优化。

## 📋 项目结构

```
YoYoFileManage/
├── datasets/              # 数据集目录
│   ├── data.yaml         # 数据集配置文件
│   ├── train/            # 训练集
│   ├── valid/            # 验证集
│   └── test/             # 测试集
├── docs/                 # 项目文档目录
│   ├── CLOUD_FILES.md    # 云服务器文件说明
│   ├── CLOUD_QUICK_START.md # 云服务器快速开始
│   ├── CLOUD_SUMMARY.md  # 云服务器总结
│   ├── CLOUD_TRAINING_GUIDE.md # 云服务器训练指南
│   ├── CONFIG_FIX_SUMMARY.md # 配置修复总结
│   ├── CONFIG_USAGE.md   # 配置使用说明
│   ├── GUIDE.md          # 使用指南
│   ├── PROJECT_STRUCTURE.md # 项目结构说明
│   ├── SUMMARY.md        # 项目总结
│   └── UNIFIED_TRAIN_GUIDE.md # 统一训练指南
├── train_yolo.py         # 主训练脚本
├── train_config.py       # 训练配置文件
├── test_yolo.py          # 模型测试脚本
├── requirements.txt      # 依赖包列表
└── README.md            # 本文件
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 开始训练

#### 💻 本地训练 (MacBook Air M1)

**方式一：直接运行（推荐新手）**
```bash
python train_yolo.py
```

**方式二：快速测试（5分钟）**
```bash
python quick_start.py
```

#### ☁️ 云服务器训练 (T4 GPU)

```bash
# 使用 T4 优化的脚本 (速度快3-5倍!)
python train_yolo_cloud.py
```

> 📖 详见 [云服务器训练指南](docs/CLOUD_TRAINING_GUIDE.md)

### 3. 测试模型

训练完成后，运行测试脚本评估模型性能：

```bash
python test_yolo.py
```

## ⚙️ 训练配置

### 默认参数 (M1 本地)

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | yolo12n.pt | 最轻量级版本 |
| Epochs | 50 | 训练轮数 |
| Batch Size | 16 | 批次大小 |
| Image Size | 640 | 图像尺寸 |
| Device | MPS | Apple Silicon加速 |
| Patience | 20 | 早停耐心值 |

### T4 GPU 配置 (云服务器)

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | yolo12s.pt | 更大模型，效果更好 |
| Epochs | 100 | 训练更多轮 |
| Batch Size | 32 | 更大批次 |
| Image Size | 640 | 图像尺寸 |
| Device | CUDA | T4 GPU 加速 |
| Cache | RAM | 缓存数据，大幅提速 |

> 📊 详细对比见 [config_comparison.py](config_comparison.py)

### 修改参数

编辑 `train_yolo.py` (M1) 或 `train_yolo_cloud.py` (T4) 中的配置部分：

```python
# 训练参数
epochs = 50              # 修改训练轮数
batch_size = 16          # 修改批次大小
img_size = 640           # 修改图像尺寸
```

### 不同场景建议

**快速测试（5分钟内）：**
```python
epochs = 5
batch_size = 8
```

**标准训练（推荐）：**
```python
epochs = 50
batch_size = 16
```

**高质量训练（追求最佳效果）：**
```python
epochs = 100
batch_size = 16
patience = 30
```

## 📊 数据集信息

- **类别数量**: 2
- **类别名称**: head, person
- **数据格式**: YOLO格式
- **数据来源**: Roboflow

## 🎯 检测目标

1. **head**: 人物头部
2. **person**: 完整人物

## 💻 M1 优化说明

本项目针对 MacBook Air M1 进行了以下优化：

1. **MPS 加速**: 使用 Apple 的 Metal Performance Shaders
2. **混合精度训练**: 启用 AMP 加速训练
3. **合理的批次大小**: 避免内存溢出
4. **优化的数据加载**: 减少 CPU 负担

### 性能预期

- **训练速度**: 约 30-120 分钟（50 epochs）
- **内存占用**: 约 4-8 GB
- **推理速度**: 约 30-60 FPS（640尺寸）

## ☁️ 云服务器 (T4 GPU)

如果需要更快的训练速度和更好的效果，推荐使用云服务器：

### T4 GPU 优势
- ⚡ **速度快 3-5倍**: 50 epochs 仅需 10-20 分钟
- 📈 **效果更好**: 可使用更大模型 (yolo12s/m)
- 🔥 **训练更久**: 可轻松训练 100-150 epochs

### 快速开始
```bash
# 1. 上传项目到服务器
scp -r YoYoFileManage/ user@server_ip:/path/

# 2. SSH 登录
ssh user@server_ip

# 3. 运行 T4 优化脚本
python train_yolo_cloud.py
```

> 📖 详细指南: [CLOUD_TRAINING_GUIDE.md](docs/CLOUD_TRAINING_GUIDE.md)  
> ⚡ 快速参考: [CLOUD_QUICK_START.md](docs/CLOUD_QUICK_START.md)

## 📈 训练结果

训练完成后，结果保存在 `runs/detect/yolo12n_person_head/` 目录下：

```
runs/detect/yolo12n_person_head/
├── weights/
│   ├── best.pt          # 最佳模型权重
│   └── last.pt          # 最后一次训练的权重
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
├── val_batch0_pred.jpg  # 验证集预测示例
└── ...
```

### 查看训练结果

训练过程中会生成以下可视化图表：

1. **results.png**: 损失、精度、召回率曲线
2. **confusion_matrix.png**: 混淆矩阵
3. **PR_curve.png**: 精度-召回率曲线
4. **F1_curve.png**: F1分数曲线

## 🔍 模型测试

### 评估模型性能

```bash
python test_yolo.py
```

这将在测试集上评估模型，输出以下指标：

- **mAP50**: 在 IoU=0.5 时的平均精度
- **mAP50-95**: 在 IoU=0.5-0.95 时的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

### 单张图片预测

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo12n_person_head/weights/best.pt')
results = model.predict('path/to/image.jpg', save=True)
```

### 批量预测

```python
model.predict('path/to/images/folder/', save=True)
```

### 视频预测

```python
model.predict('path/to/video.mp4', save=True)
```

## 🔧 故障排除

### 1. MPS 不可用

如果看到 "MPS不可用" 的提示：
- 确保 macOS >= 12.3
- 确保 PyTorch >= 2.0
- 尝试使用 CPU: 修改 `device = 'cpu'`

### 2. 内存不足

如果训练时内存溢出：
```python
batch_size = 8   # 降低批次大小
workers = 2      # 减少数据加载线程
```

### 3. 训练速度慢

- 确认 MPS 已启用
- 减少数据增强参数
- 降低图像尺寸（如 320）

### 4. 模型文件下载失败

首次运行会自动下载预训练权重，如果失败：
- 检查网络连接
- 使用代理
- 手动下载后放到相应目录

## 📝 下一步计划

### 如果效果满意：
1. ✅ 可以直接在 M1 上部署使用
2. ✅ 导出为其他格式（ONNX、CoreML）
3. ✅ 优化推理速度

### 如果需要提升性能：
1. 🔄 增加训练轮数（100 epochs）
2. 🔄 使用更大模型（yolo12s 或 yolo12m）
3. 🔄 增加数据增强
4. 🔄 采集更多训练数据

### 如果需要更快训练：
1. 🚀 使用 CUDA GPU（需要 NVIDIA 显卡）
2. 🚀 使用云端 GPU（Google Colab、AWS等）

## 📚 参考资料

- [Ultralytics YOLOv12 文档](https://docs.ultralytics.com/)
- [YOLO 训练技巧](https://docs.ultralytics.com/guides/training-tips/)
- [模型导出指南](https://docs.ultralytics.com/modes/export/)

## 📄 License

本项目遵循数据集的许可协议: CC BY 4.0

## 🤝 贡献

欢迎提出问题和建议！

---

**祝您训练顺利！** 🎉
