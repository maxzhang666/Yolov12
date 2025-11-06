# 🎯 统一训练脚本使用指南

## ✅ 问题解决

您的质疑**完全正确**！原来的设计确实有问题：
- ❌ 没有明确的方式选择 M1 还是 T4 配置
- ❌ 需要手动编辑代码来切换配置
- ❌ 不够灵活

现在创建了 **`train.py` 统一训练脚本** ⭐，完美解决了这个问题！

---

## 🚀 使用方式

### 方式1: 使用预设配置（推荐）⭐

#### M1 配置

```bash
# 快速测试 (5 epochs, ~5分钟)
python train.py --config m1_quick

# 标准训练 (50 epochs, ~30-60分钟) ⭐ 推荐
python train.py --config m1_standard

# M1 优化配置 (内存友好)
python train.py --config m1_optimized
```

#### T4 配置

```bash
# 快速测试 (10 epochs, yolo12n, ~10分钟)
python train.py --config t4_quick

# 标准训练 (100 epochs, yolo12s, ~30-50分钟) ⭐ 推荐
python train.py --config t4_standard

# 高质量训练 (150 epochs, yolo12m, ~60-90分钟)
python train.py --config t4_highquality
```

---

### 方式2: 命令行参数自定义

```bash
# M1 自定义
python train.py --device mps --model yolo12n.pt --epochs 80 --batch 12

# T4 自定义
python train.py --device cuda --model yolo12s.pt --epochs 120 --batch 32 --cache ram

# CPU 训练（备用）
python train.py --device cpu --model yolo12n.pt --epochs 10 --batch 4
```

---

### 方式3: 混合使用

```bash
# 使用预设配置，但修改某些参数
python train.py --config m1_standard --epochs 80
python train.py --config t4_standard --batch 24
```

---

## 📊 预设配置详情

### M1 配置

| 配置名 | 模型 | Epochs | Batch | 时间 | 用途 |
|--------|------|--------|-------|------|------|
| `m1_quick` | yolo12n | 5 | 8 | ~5分钟 | 快速测试 |
| `m1_standard` ⭐ | yolo12n | 50 | 16 | ~30-60分钟 | 标准训练 |
| `m1_optimized` | yolo12n | 50 | 8 | ~40-80分钟 | 内存优化 |

### T4 配置

| 配置名 | 模型 | Epochs | Batch | Cache | 时间 | 用途 |
|--------|------|--------|-------|-------|------|------|
| `t4_quick` | yolo12n | 10 | 48 | RAM | ~10分钟 | 快速测试 |
| `t4_standard` ⭐ | yolo12s | 100 | 32 | RAM | ~30-50分钟 | 标准训练 |
| `t4_highquality` | yolo12m | 150 | 24 | RAM | ~60-90分钟 | 高质量 |

---

## 💡 命令行参数说明

### 核心参数

```bash
--config     预设配置名称
--model      模型文件 (yolo12n.pt / yolo12s.pt / yolo12m.pt)
--epochs     训练轮数
--batch      批次大小
--device     设备 (mps / cuda / cpu / 0 / 1)
--workers    数据加载线程数
--cache      缓存方式 (ram / disk / false)
```

### 使用示例

```bash
# 查看所有参数
python train.py --help

# 使用预设配置
python train.py --config m1_standard

# 自定义所有参数
python train.py \
  --model yolo12s.pt \
  --epochs 100 \
  --batch 24 \
  --device cuda \
  --workers 8 \
  --cache ram
```

---

## 🔍 配置对比

### 本地 M1 vs 云端 T4

```bash
# M1 标准配置
python train.py --config m1_standard
# 输出: device=mps, model=yolo12n, epochs=50, batch=16

# T4 标准配置
python train.py --config t4_standard
# 输出: device=cuda, model=yolo12s, epochs=100, batch=32, cache=ram
```

**关键区别：**
- 设备: `mps` vs `cuda`
- 模型: `yolo12n` vs `yolo12s`
- Epochs: `50` vs `100`
- Batch: `16` vs `32`
- Cache: 无 vs `ram`

---

## 📝 实际使用场景

### 场景1: 本地快速测试
```bash
# M1 上快速验证（5分钟）
python train.py --config m1_quick
```

### 场景2: 本地标准训练
```bash
# M1 上正式训练（30-60分钟）
python train.py --config m1_standard
```

### 场景3: 云端快速测试
```bash
# T4 上快速验证（10分钟）
python train.py --config t4_quick
```

### 场景4: 云端正式训练
```bash
# T4 上正式训练（30-50分钟）
python train.py --config t4_standard
```

### 场景5: 追求最佳效果
```bash
# T4 上高质量训练（60-90分钟）
python train.py --config t4_highquality
```

### 场景6: 自定义实验
```bash
# 自定义配置
python train.py --device cuda --model yolo12m.pt --epochs 200 --batch 16 --cache ram
```

---

## 🎯 推荐工作流程

### 第1步: 本地快速验证
```bash
# 在 M1 上快速测试（5分钟）
python train.py --config m1_quick
```

### 第2步: 选择训练平台

**如果本地训练即可：**
```bash
python train.py --config m1_standard
```

**如果需要更好效果：**
```bash
# 上传到云服务器
scp -r YoYoFileManage/ user@server:/path/

# SSH 登录
ssh user@server
cd YoYoFileManage

# T4 训练
python train.py --config t4_standard
```

---

## ✨ 优势总结

### ✅ 解决了原来的问题
1. ✅ **明确指定配置**：通过 `--config` 参数
2. ✅ **M1 和 T4 都支持**：预设了所有配置
3. ✅ **无需编辑代码**：命令行参数控制
4. ✅ **灵活自定义**：可以覆盖任何参数

### ✅ 新的优势
1. ✅ **一个入口**：`train.py` 统一所有训练
2. ✅ **自动检测**：自动检测设备可用性
3. ✅ **清晰显示**：启动时显示所有配置
4. ✅ **易于使用**：简单的命令行参数

---

## 📋 文件对比

| 文件 | 用途 | 何时使用 |
|------|------|---------|
| **train.py** ⭐ | 统一训练入口 | **推荐使用** |
| `train_yolo.py` | M1 专用（旧） | 已被 train.py 替代 |
| `train_yolo_cloud.py` | T4 专用（旧） | 已被 train.py 替代 |
| `train_config.py` | 配置定义 | 被 train.py 使用 |

**推荐：直接使用 `train.py`！**

---

## 🆚 新旧对比

### ❌ 旧方式（不推荐）

```bash
# M1 训练
python train_yolo.py

# T4 训练
python train_yolo_cloud.py

# 切换配置？需要编辑代码...
```

### ✅ 新方式（推荐）⭐

```bash
# M1 训练
python train.py --config m1_standard

# T4 训练
python train.py --config t4_standard

# 快速测试
python train.py --config m1_quick

# 自定义
python train.py --device cuda --epochs 150
```

---

## 🎉 总结

**感谢您的反馈！** 🙏

现在：
1. ✅ 创建了 **`train.py` 统一训练脚本**
2. ✅ 通过 `--config` 参数**明确选择** M1 或 T4 配置
3. ✅ 支持**命令行参数自定义**
4. ✅ **无需编辑代码**即可切换配置

**推荐使用：`train.py`**

```bash
# M1 本地训练
python train.py --config m1_standard

# T4 云端训练
python train.py --config t4_standard
```

简单、清晰、灵活！✨
