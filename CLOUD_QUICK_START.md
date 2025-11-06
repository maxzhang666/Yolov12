# ☁️ 云服务器快速迁移指南

## 🎯 一句话总结

**只需要修改 5 个参数，其他都已优化好！**

---

## ⚡ 核心调整（必改）

### 1. 设备类型
```python
# M1 版本
device = 'mps'

# T4 版本 ✅
device = 0  # 或 'cuda'
```

### 2. 模型大小
```python
# M1 版本
model_name = 'yolo12n.pt'

# T4 版本 ✅ (推荐)
model_name = 'yolo12s.pt'  # 效果更好
```

### 3. Batch Size
```python
# M1 版本
batch_size = 16

# T4 版本 ✅
batch_size = 32  # T4显存够用
```

### 4. Workers
```python
# M1 版本
workers = 4

# T4 版本 ✅
workers = 8  # 8核CPU
```

### 5. 缓存数据
```python
# M1 版本
# (没有这个参数)

# T4 版本 ✅ (新增)
cache = 'ram'  # 大幅提速！
```

---

## 📂 文件对应关系

| 用途 | M1 本地 | T4 云端 |
|------|---------|---------|
| **训练脚本** | `train_yolo.py` | `train_yolo_cloud.py` ⭐ |
| **测试脚本** | `test_yolo.py` | `test_yolo.py` (通用) |
| **监控工具** | `monitor.py` | `monitor_cloud.py` ⭐ |
| **配置对比** | - | `config_comparison.py` ⭐ |
| **使用指南** | `GUIDE.md` | `CLOUD_TRAINING_GUIDE.md` ⭐ |

---

## 🚀 云服务器使用步骤（5步）

### Step 1: 上传项目
```bash
# 打包项目（在本地M1上）
cd /Users/maxzhang/PycharmProjects
tar -czf YoYoFileManage.tar.gz YoYoFileManage/

# 上传到服务器
scp YoYoFileManage.tar.gz user@server_ip:/home/user/

# SSH登录服务器
ssh user@server_ip

# 解压
tar -xzf YoYoFileManage.tar.gz
cd YoYoFileManage
```

### Step 2: 安装环境
```bash
# 安装依赖
pip install -r requirements.txt

# 验证CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: 开始训练
```bash
# 使用 T4 优化的脚本 ⭐
python train_yolo_cloud.py

# 或后台运行（推荐）
nohup python train_yolo_cloud.py > training.log 2>&1 &
```

### Step 4: 监控训练
```bash
# 新开一个SSH窗口，监控训练+GPU
python monitor_cloud.py watch

# 或只监控GPU
watch -n 1 nvidia-smi
```

### Step 5: 下载结果
```bash
# 训练完成后，在本地M1上执行
scp user@server_ip:/home/user/YoYoFileManage/runs/detect/yolo12s_person_head_t4/weights/best.pt ./
```

---

## 📊 参数对比速查表

| 参数 | M1 | T4 | 说明 |
|------|----|----|------|
| **device** | `'mps'` | `0` | 设备类型 |
| **model** | `yolo12n.pt` | `yolo12s.pt` | 模型大小 |
| **epochs** | `50` | `100` | 训练轮数 |
| **batch_size** | `16` | `32` | 批次大小 |
| **workers** | `4` | `8` | CPU线程 |
| **cache** | - | `'ram'` | 数据缓存 |
| **optimizer** | `'auto'` | `'AdamW'` | 优化器 |
| **lr0** | `0.01` | `0.001` | 学习率 |
| **mixup** | `0.0` | `0.15` | 数据增强 |
| **patience** | `20` | `30` | 早停 |
| **save_period** | `10` | `20` | 保存频率 |

---

## 🎯 不同场景的配置

### 场景A: 快速验证（10分钟）
```python
# 使用 train_yolo_cloud.py，临时修改:
epochs = 10
batch_size = 48
model_name = 'yolo12n.pt'
```

### 场景B: 标准训练（30-50分钟）⭐ 推荐
```python
# 使用 train_yolo_cloud.py (默认配置)
epochs = 100
batch_size = 32
model_name = 'yolo12s.pt'
cache = 'ram'
```

### 场景C: 追求最佳效果（1-2小时）
```python
# 修改 train_yolo_cloud.py:
epochs = 150
batch_size = 24
model_name = 'yolo12m.pt'
cache = 'ram'
patience = 50
```

### 场景D: 显存不足时
```python
# 修改 train_yolo_cloud.py:
batch_size = 16  # 降低
workers = 4      # 降低
cache = False    # 关闭
model_name = 'yolo12n.pt'  # 用小模型
```

---

## 📋 检查清单

### 上传前检查
- [ ] 数据集 `datasets/` 完整
- [ ] `data.yaml` 配置正确
- [ ] 已在本地M1测试通过

### 服务器上检查
- [ ] `nvidia-smi` 显示 T4 GPU
- [ ] Python 版本 >= 3.8
- [ ] PyTorch CUDA 可用
- [ ] 所有依赖已安装

### 训练时检查
- [ ] GPU 利用率 > 80%
- [ ] 显存使用正常（不是100%）
- [ ] loss 正常下降
- [ ] 使用 tmux/screen 防断连

### 训练后检查
- [ ] 已下载 `best.pt`
- [ ] 已下载 `results.png`
- [ ] 已下载训练日志
- [ ] 已备份重要文件

---

## 💡 常用命令

### 查看GPU
```bash
nvidia-smi                    # 查看GPU状态
watch -n 1 nvidia-smi         # 实时监控
nvidia-smi -l 1               # 每秒刷新
```

### 后台训练
```bash
# 方式1: nohup
nohup python train_yolo_cloud.py > training.log 2>&1 &

# 方式2: tmux (推荐)
tmux new -s yolo
python train_yolo_cloud.py
# 按 Ctrl+B, D 断开
tmux attach -s yolo  # 重新连接

# 方式3: screen
screen -S yolo
python train_yolo_cloud.py
# 按 Ctrl+A, D 断开
screen -r yolo  # 重新连接
```

### 查看日志
```bash
tail -f training.log          # 实时查看
tail -100 training.log        # 查看最后100行
grep "Epoch" training.log     # 搜索特定内容
```

### 下载文件
```bash
# 下载模型
scp user@server:/path/to/best.pt ./

# 下载整个目录
scp -r user@server:/path/to/runs ./

# 使用rsync (更快)
rsync -avz user@server:/path/to/runs ./
```

---

## ⚠️ 常见问题

### Q1: CUDA Out of Memory
```python
# 解决: 降低这些参数
batch_size = 16  # 从32降到16
workers = 4
cache = False
```

### Q2: GPU利用率低（<50%）
```python
# 可能原因:
1. workers 太少 → 增加到8
2. 没缓存数据 → cache='ram'
3. batch_size 太小 → 增加到32
```

### Q3: 训练中断怎么办
```bash
# 继续训练
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/*/weights/last.pt')
model.train(resume=True)
"
```

### Q4: 无法连接服务器
```bash
# 检查网络
ping server_ip

# 检查SSH
ssh -v user@server_ip

# 使用密钥登录
ssh -i ~/.ssh/id_rsa user@server_ip
```

---

## 🎓 最佳实践

### 1. 先本地测试，再云端训练
```bash
# 本地M1: 快速验证（5分钟）
python quick_start.py

# 云端T4: 正式训练（30-50分钟）
python train_yolo_cloud.py
```

### 2. 使用 tmux 防止断连
```bash
# 创建会话
tmux new -s yolo_train

# 运行训练
python train_yolo_cloud.py

# 断开（训练继续）: Ctrl+B, D
# 重新连接
tmux attach -t yolo_train
```

### 3. 监控训练进度
```bash
# 终端1: 训练
python train_yolo_cloud.py

# 终端2: 监控
python monitor_cloud.py watch

# 终端3: GPU监控
watch -n 1 nvidia-smi
```

### 4. 及时下载结果
```bash
# 训练完立即下载，避免数据丢失
scp -r user@server:/path/to/runs ./backup/
```

---

## 📈 性能预期

### 训练时间
| 配置 | M1 | T4 | 加速比 |
|------|----|----|--------|
| yolo12n, 50 epochs | 30-60分钟 | 10-15分钟 | **3-5x** |
| yolo12s, 100 epochs | 不推荐 | 30-50分钟 | - |
| yolo12m, 150 epochs | 不推荐 | 60-90分钟 | - |

### 模型效果
| 模型 | M1 (50e) | T4 (100e) | 提升 |
|------|----------|-----------|------|
| yolo12n | 0.50-0.60 | 0.52-0.62 | +2-5% |
| yolo12s | - | 0.55-0.65 | - |
| yolo12m | - | 0.60-0.70 | - |

---

## ✅ 总结

### 您需要做的：
1. ✅ 上传项目到服务器
2. ✅ 运行 `train_yolo_cloud.py`
3. ✅ 监控训练进度
4. ✅ 下载训练好的模型

### 已经为您准备好的：
1. ✅ T4 优化的训练脚本
2. ✅ GPU 监控工具
3. ✅ 详细使用文档
4. ✅ 配置对比参考

---

**🎉 现在您可以开始在 T4 上训练了！**

相关文件：
- 训练: `train_yolo_cloud.py`
- 监控: `monitor_cloud.py`
- 对比: `config_comparison.py`
- 文档: `CLOUD_TRAINING_GUIDE.md`
