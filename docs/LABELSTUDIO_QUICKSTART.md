# YOLO 标注数据二次审核工具

## 快速开始

本工具可以将 YOLO 格式的标注数据转换为 Label Studio 格式，方便进行二次审核和微调。

### 1. 安装依赖

```bash
pip install pillow pyyaml
```

### 2. 使用方法

#### 方法一：使用命令行（推荐）

```bash
# 转换测试集（最常用于审核）
python3 yolo2label_studio.py --dataset test --output test_review.json

# 转换训练集
python3 yolo2label_studio.py --dataset train --output train_review.json

# 转换验证集
python3 yolo2label_studio.py --dataset valid --output valid_review.json
```

#### 方法二：使用交互式脚本

```bash
./convert_to_labelstudio.sh
```

然后按照提示选择要转换的数据集。

### 3. 导入到 Label Studio

1. **安装 Label Studio**（如果还没有安装）：
   ```bash
   pip install label-studio
   label-studio start
   ```

2. **创建项目**：
   - 访问 http://localhost:8080
   - 点击 "Create Project"
   - 输入项目名称

3. **配置标注界面**：
   - 在 "Labeling Setup" 中，选择 "Custom Template"
   - 复制 `label_studio_config.xml` 的内容并粘贴
   - 点击 "Save"

4. **导入数据**：
   - 点击 "Import" 按钮
   - 选择生成的 JSON 文件（如 `test_review.json`）
   - 点击 "Import"

5. **开始审核**：
   - 点击任务开始查看和修改标注
   - 使用快捷键提高效率：
     - `1` - body (红色)
     - `2` - head (绿色)  
     - `3` - leg (蓝色)
     - `Delete` - 删除标注
     - `Ctrl+Z` - 撤销

### 4. 文件说明

- `yolo2label_studio.py` - 主转换脚本
- `convert_to_labelstudio.sh` - 交互式批量转换脚本
- `label_studio_config.xml` - Label Studio 配置模板
- `docs/YOLO_TO_LABELSTUDIO.md` - 详细使用文档

### 5. 常见使用场景

#### 场景1：快速审核测试集

```bash
python3 yolo2label_studio.py --dataset test --output test_review.json
```

#### 场景2：审核特定目录的数据

```bash
python3 yolo2label_studio.py \
    --dataset-path ./datasets/problematic_images \
    --output problematic_review.json
```

#### 场景3：批量转换所有数据

```bash
./convert_to_labelstudio.sh
# 选择选项 4
```

### 6. 输出示例

运行转换脚本后，您会看到类似以下的输出：

```
Loading config from: datasets/data.yaml
Classes: ['body', 'head', 'leg']
Dataset path: datasets/test
Images directory: datasets/test/images
Labels directory: datasets/test/labels

Starting conversion...
Found 234 images in datasets/test/images
Processed 100 images...
Processed 200 images...
Successfully converted 234 images

✓ Successfully converted 234 tasks
✓ Output saved to: test_review.json

You can now import test_review.json into Label Studio

Statistics:
  Total tasks: 234
  Total annotations: 1856
  Average annotations per image: 7.93
```

### 7. 下一步

审核完成后：

1. 从 Label Studio 导出标注数据（JSON 格式）
2. 使用相应的脚本将 Label Studio 格式转回 YOLO 格式
3. 使用修正后的数据重新训练模型

### 8. 故障排查

**问题：找不到图像**
- 确保 `datasets/` 目录结构正确
- 检查 `data.yaml` 配置

**问题：Label Studio 无法显示图像**
- 检查图像路径是否正确
- 考虑将图像复制到 Label Studio 的 media 目录

**问题：类别不匹配**
- 确保 `label_studio_config.xml` 中的类别与 `data.yaml` 一致

### 9. 获取帮助

```bash
# 查看命令行帮助
python3 yolo2label_studio.py --help

# 查看详细文档
cat docs/YOLO_TO_LABELSTUDIO.md
```

### 10. 项目数据集信息

当前项目的类别：
- `body` - 身体
- `head` - 头部
- `leg` - 腿部

数据集统计：
- 训练集：`datasets/train/`
- 验证集：`datasets/valid/`
- 测试集：`datasets/test/`

---

**提示**：建议先从测试集开始审核，因为测试集通常数量较少，可以快速验证标注质量。
