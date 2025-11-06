# ☁️ 云服务器训练指南 (T4 GPU)

## 🚀 云服务器配置

**您的配置：**
- CPU: 8核
- GPU: NVIDIA T4 16GB
- 内存: 16GB
- 系统: 通常是 Ubuntu Linux

---

## 📊 M1 vs T4 对比

| 项目 | MacBook Air M1 | 云服务器 T4 | 提升 |
|------|---------------|------------|-----|
| **设备** | MPS | CUDA | - |
| **模型推荐** | yolo12n | yolo12s/m | ⬆️ |
| **Batch Size** | 8-16 | 24-32 | 2x |
| **Workers** | 4 | 8 | 2x |
| **Epochs推荐** | 50 | 100-150 | 2-3x |
| **训练时间(50e)** | 30-60分钟 | 10-20分钟 | **3-5x** ⚡ |
| **缓存数据** | ❌ | ✅ RAM | ⚡ |
| **混合精度** | ✅ | ✅ | - |

---

## 🔄 需要调整的参数

### 1️⃣ 核心参数调整

```python
# M1 配置 → T4 配置

# 设备
device = 'mps'           →  device = 0  # 或 'cuda'

# 模型（可以用更大的）
model_name = 'yolo12n.pt'  →  model_name = 'yolo12s.pt'
                              # 或 'yolo12m.pt' (更好效果)

# Batch Size（显存允许可以更大）
batch_size = 16          →  batch_size = 32
                              # T4 16GB: 32-48 (yolo12s)
                              # 如果yolo12m: 24-32

# Epochs（训练更快可以多训练）
epochs = 50              →  epochs = 100
                              # 或 150 (追求更好效果)

# Workers（CPU核心数）
workers = 4              →  workers = 8

# 缓存（16GB内存够用）
# 新增参数
cache = 'ram'                 # 缓存到内存，大幅提速！
```

### 2️⃣ 优化器调整（可选）

```python
# T4 上可以用更激进的优化器
optimizer = 'auto'       →  optimizer = 'AdamW'
                              # AdamW 通常收敛更快

# 学习率
lr0 = 0.01               →  lr0 = 0.001
                              # AdamW 用较小学习率

# 学习率衰减
# 新增
cos_lr = True                 # 余弦学习率衰减
```

### 3️⃣ 数据增强增强（可选）

```python
# T4 训练快，可以用更强的增强
mixup = 0.0              →  mixup = 0.15
                              # 启用 mixup 增强
```

### 4️⃣ 其他优化

```python
# 保存频率（训练快了可以少保存）
save_period = 10         →  save_period = 20

# 早停耐心值
patience = 20            →  patience = 30

# 确定性训练（关闭可提速）
deterministic = True     →  deterministic = False
```

---

## 📝 完整对比示例

### M1 配置 (train_yolo.py)

```python
model_name = 'yolo12n.pt'
epochs = 50
batch_size = 16
device = 'mps'
workers = 4
cache = False  # 隐式
```

### T4 配置 (train_yolo_cloud.py) ⭐

```python
model_name = 'yolo12s.pt'  # 更大模型
epochs = 100               # 训练更多轮
batch_size = 32            # 更大批次
device = 0                 # CUDA GPU
workers = 8                # 8核CPU
cache = 'ram'              # 缓存到内存 ⚡
optimizer = 'AdamW'        # 更快收敛
cos_lr = True              # 余弦学习率
mixup = 0.15               # 混合增强
```

---

## 🎯 推荐配置方案

### 方案A: 平衡方案（推荐）⭐

```python
model_name = 'yolo12s.pt'
epochs = 100
batch_size = 32
device = 0
workers = 8
cache = 'ram'
```

**特点：**
- 训练时间: ~30-50分钟
- 效果: 优于 M1 上的 yolo12n
- 显存占用: ~8-10GB
- 适合: 大多数场景

### 方案B: 快速测试

```python
model_name = 'yolo12n.pt'
epochs = 50
batch_size = 48
device = 0
workers = 8
cache = 'ram'
```

**特点：**
- 训练时间: ~10-15分钟
- 效果: 与 M1 相当
- 适合: 快速验证

### 方案C: 追求最佳效果

```python
model_name = 'yolo12m.pt'  # 更大模型
epochs = 150
batch_size = 24
device = 0
workers = 8
cache = 'ram'
```

**特点：**
- 训练时间: ~1-2小时
- 效果: 最佳
- 显存占用: ~12-14GB
- 适合: 追求性能

---

## 🚀 云服务器使用步骤

### 1. 上传项目到云服务器

```bash
# 方式1: 使用 scp
scp -r YoYoFileManage/ user@server_ip:/path/to/

# 方式2: 使用 git
git clone <your_repo_url>

# 方式3: 使用 rsync
rsync -avz YoYoFileManage/ user@server_ip:/path/to/
```

### 2. 安装环境

```bash
# SSH登录服务器
ssh user@server_ip

# 进入项目目录
cd YoYoFileManage

# 安装依赖（如果没有conda）
pip install -r requirements.txt

# 或使用conda（推荐）
conda create -n yolo python=3.10
conda activate yolo
pip install -r requirements.txt
```

### 3. 验证CUDA

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. 开始训练

```bash
# 使用T4优化的脚本
python train_yolo_cloud.py

# 或在后台运行（推荐）
nohup python train_yolo_cloud.py > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### 5. 监控训练

```bash
# 方式1: 使用 nvidia-smi 监控GPU
watch -n 1 nvidia-smi

# 方式2: 使用我们的监控工具
python monitor.py watch

# 方式3: 查看日志
tail -f training.log
```

### 6. 下载结果

```bash
# 训练完成后，下载模型
scp user@server_ip:/path/to/runs/detect/*/weights/best.pt ./
```

---

## ⚡ 性能优化技巧

### 1. 缓存数据集 ⭐⭐⭐

```python
cache = 'ram'  # 16GB内存够用，大幅提速！
```

**效果：** 可提速 **30-50%**

### 2. 调整 Batch Size

```python
# 测试最大可用 batch size
# 从小到大尝试: 16 → 24 → 32 → 40 → 48
batch_size = 32  # 找到不OOM的最大值
```

**规律：**
- yolo12n: 可用 48-64
- yolo12s: 可用 32-40
- yolo12m: 可用 24-32

### 3. 混合精度训练 ⭐⭐⭐

```python
amp = True  # 默认已启用
```

**效果：** 提速 **20-40%**，节省显存

### 4. 多GPU训练（如果有多个GPU）

```python
device = [0, 1]  # 使用2个GPU
```

### 5. 优化数据加载

```python
workers = 8      # 设为CPU核心数
persistent_workers = True  # 保持worker进程
```

---

## 📊 预期性能表现

### 训练时间对比

| 配置 | M1 (50 epochs) | T4 (50 epochs) | T4 (100 epochs) |
|------|----------------|----------------|-----------------|
| yolo12n | 30-60分钟 | 10-15分钟 | 20-30分钟 |
| yolo12s | 不推荐 | 15-25分钟 | 30-50分钟 |
| yolo12m | 不推荐 | 30-45分钟 | 60-90分钟 |

### 模型性能预期

| 模型 | mAP50-95 预期 | 推理速度 (T4) | 模型大小 |
|------|--------------|---------------|---------|
| yolo12n | 0.50-0.60 | ~2ms/img | ~6MB |
| yolo12s | 0.55-0.65 | ~3ms/img | ~12MB |
| yolo12m | 0.60-0.70 | ~5ms/img | ~26MB |

---

## 🔧 常见问题

### Q1: CUDA Out of Memory

**解决方案：**
```python
batch_size = 24  # 或 16
workers = 4      # 减少worker
cache = False    # 关闭缓存
```

### Q2: 训练速度没提升

**检查清单：**
- ✅ 确认使用 `device=0` 不是 `device='cpu'`
- ✅ 确认 `cache='ram'` 已启用
- ✅ 确认 `amp=True` 已启用
- ✅ 运行 `nvidia-smi` 确认GPU在使用

### Q3: 数据集缓存失败

```python
# 如果内存不够缓存
cache = 'disk'  # 缓存到磁盘
# 或
cache = False   # 关闭缓存
```

### Q4: 连接断开训练中断

**使用 tmux 或 screen：**
```bash
# 创建会话
tmux new -s yolo_train

# 运行训练
python train_yolo_cloud.py

# 断开会话（训练继续）: Ctrl+B, D

# 重新连接
tmux attach -t yolo_train
```

---

## 📋 云服务器检查清单

### 训练前
- [ ] 已上传完整项目（包括 datasets/）
- [ ] 已安装所有依赖
- [ ] CUDA 可用（`nvidia-smi`）
- [ ] PyTorch 检测到 GPU
- [ ] 数据集路径正确
- [ ] 已修改参数（device, batch_size等）

### 训练中
- [ ] GPU利用率 >80%（`nvidia-smi`）
- [ ] 显存使用合理（不是100%）
- [ ] loss 正常下降
- [ ] 使用 tmux/screen 防止断连

### 训练后
- [ ] 下载 best.pt 模型
- [ ] 下载 results.png 查看曲线
- [ ] 下载 results.csv 数据
- [ ] 备份训练日志

---

## 💾 文件传输命令

### 下载模型到本地

```bash
# 下载最佳模型
scp user@server_ip:/path/to/runs/detect/yolo12s_person_head_t4/weights/best.pt ./models/

# 下载整个结果目录
scp -r user@server_ip:/path/to/runs/detect/yolo12s_person_head_t4/ ./results/

# 下载训练日志
scp user@server_ip:/path/to/training.log ./logs/
```

---

## 🎓 最佳实践建议

### 1. 先在 T4 上快速测试
```bash
# 使用小模型快速验证（10分钟）
python train_yolo_cloud.py  # 临时改为 epochs=10
```

### 2. 确认无误后正式训练
```bash
# 后台运行，防止断连
nohup python train_yolo_cloud.py > training.log 2>&1 &
```

### 3. 定期检查
```bash
# 每10分钟查看一次
tail -f training.log
nvidia-smi
```

### 4. 训练完成后立即下载
```bash
# 避免云服务器数据丢失
scp -r user@server:/path/to/runs ./backup/
```

---

## 📊 总结对比表

| 项目 | 本地 M1 | 云服务器 T4 | 推荐 |
|------|---------|------------|-----|
| **成本** | 免费 | 付费 | - |
| **速度** | 慢 | **快3-5倍** | T4 ⭐ |
| **模型** | yolo12n | yolo12s/m | T4 ⭐ |
| **效果** | 一般 | **更好** | T4 ⭐ |
| **便利性** | 方便 | 需要上传 | M1 |
| **适合** | 测试、小数据 | 正式训练 | 各有优势 |

---

## 🚀 推荐工作流程

1. **本地 M1 快速原型**
   - 用 `quick_start.py` 测试（5分钟）
   - 验证数据集没问题
   - 验证代码可运行

2. **云服务器 T4 正式训练**
   - 上传项目到 T4
   - 使用 `train_yolo_cloud.py`
   - 训练 100-150 epochs
   - 获得高质量模型

3. **本地 M1 部署使用**
   - 下载训练好的模型
   - 在本地进行推理
   - 实际应用部署

---

**您已经有完整的训练脚本了！** 🎉

- M1 本地: `train_yolo.py`
- T4 云端: `train_yolo_cloud.py` ⭐ (新创建)

直接使用即可！
