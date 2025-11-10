# YOLO to Label Studio 转换工具使用指南

## 📋 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [使用方法](#使用方法)
- [Label Studio 导入](#label-studio-导入)
- [输出格式说明](#输出格式说明)
- [常见问题](#常见问题)
- [完整工作流程](#完整工作流程)

## 概述

`yolo2label_studio.py` 是一个将YOLO格式的标注数据转换为Label Studio导入格式的工具，方便对已标注的数据进行二次审核和微调。

### 功能特性

- ✅ 支持标准YOLO格式（归一化坐标）
- ✅ 自动读取类别配置
- ✅ 支持train/valid/test三个数据集分割
- ✅ 支持自定义数据集路径
- ✅ 保留原始标注信息
- ✅ 生成Label Studio标准JSON格式
- ✅ 相对路径支持（相对于datasets文件夹）
- ✅ 提供详细的转换统计信息

## 快速开始

### 1. 安装依赖

```bash
pip install pillow pyyaml
```

### 2. 转换数据集

**方式一：命令行（推荐）**

```bash
# 转换测试集（最常用于审核）
python3 yolo2label_studio.py --dataset test --output test_review.json

# 转换训练集
python3 yolo2label_studio.py --dataset train --output train_review.json

# 转换验证集
python3 yolo2label_studio.py --dataset valid --output valid_review.json
```

**方式二：交互式脚本**

```bash
./convert_to_labelstudio.sh
```

然后按照提示选择要转换的数据集。

### 3. 查看输出

转换成功后，您会看到：

```
Loading config from: datasets/data.yaml
Classes: ['body', 'head', 'leg']
Dataset path: datasets/test
Images directory: datasets/test/images
Labels directory: datasets/test/labels
Relative path for Label Studio: test/images

Starting conversion...
Found 503 images in datasets/test/images
Processed 99 images...
Processed 199 images...
Processed 299 images...
Processed 399 images...
Processed 499 images...
Successfully converted 503 images

✓ Successfully converted 503 tasks
✓ Output saved to: test_review.json

You can now import test_review.json into Label Studio

Statistics:
  Total tasks: 503
  Total annotations: 698
  Average annotations per image: 1.39
```

## 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--output` | 输出的JSON文件路径 | `--output test_review.json` |

### 数据集选择（二选一）

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--dataset` | 标准数据集分割 | `train`, `valid`, `test` |
| `--dataset-path` | 自定义数据集路径 | 任意路径 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `./datasets/data.yaml` | 配置文件路径 |
| `--project-root` | `./datasets` | 项目根目录 |

### 完整示例

```bash
# 基本使用
python3 yolo2label_studio.py --dataset test --output output.json

# 指定所有参数
python3 yolo2label_studio.py \
    --dataset train \
    --config ./datasets/data.yaml \
    --project-root ./datasets \
    --output train_labelstudio.json
```

## 使用方法

### 场景1：审核测试集

最常见的使用场景，快速审核测试集的标注质量：

```bash
python3 yolo2label_studio.py --dataset test --output test_review.json
```

### 场景2：审核所有数据集

使用批量转换脚本：

```bash
./convert_to_labelstudio.sh
# 选择选项 4（全部转换）
```

或手动执行：

```bash
python3 yolo2label_studio.py --dataset train --output train_ls.json
python3 yolo2label_studio.py --dataset valid --output valid_ls.json
python3 yolo2label_studio.py --dataset test --output test_ls.json
```

### 场景3：审核特定目录

如果您有一些问题图像需要单独审核：

```bash
python3 yolo2label_studio.py \
    --dataset-path ./datasets/problem_images \
    --output problem_images_review.json
```

## 目录结构要求

```
datasets/
├── data.yaml           # 配置文件
├── train/
│   ├── images/        # 训练图像
│   └── labels/        # 训练标注
├── valid/
│   ├── images/        # 验证图像
│   └── labels/        # 验证标注
└── test/
    ├── images/        # 测试图像
    └── labels/        # 测试标注
```

## data.yaml 配置格式

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: ['body', 'head', 'leg']
```

## 输出格式说明

### JSON 结构

生成的JSON文件符合Label Studio标准格式：

```json
[
  {
    "data": {
      "image": "/data/local-files/?d=test/images/image_001.jpg"
    },
    "annotations": [
      {
        "model_version": "yolo_converted",
        "result": [
          {
            "id": "1_0",
            "type": "rectanglelabels",
            "value": {
              "x": 45.2,
              "y": 30.5,
              "width": 10.3,
              "height": 15.2,
              "rotation": 0,
              "rectanglelabels": ["head"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 640,
            "original_height": 640
          }
        ]
      }
    ]
  }
]
```

### 字段说明

#### 图像路径格式

```json
"image": "/data/local-files/?d=test/images/filename.jpg"
```

- 前缀：`/data/local-files/?d=`（Label Studio本地文件格式）
- 路径：相对于 `datasets` 文件夹的相对路径
- 格式：`数据集/images/文件名.jpg`

#### 标注格式

坐标系统：
- **x, y**: 边界框左上角坐标（百分比 0-100）
- **width, height**: 边界框宽高（百分比 0-100）
- **rotation**: 旋转角度（通常为0）

### 目录结构要求

```
datasets/
├── data.yaml           # 配置文件
├── train/
│   ├── images/        # 训练图像
│   └── labels/        # 训练标注（.txt格式）
├── valid/
│   ├── images/        # 验证图像
│   └── labels/        # 验证标注
└── test/
    ├── images/        # 测试图像
    └── labels/        # 测试标注
```

### data.yaml 配置示例

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: ['body', 'head', 'leg']

roboflow:
  workspace: your_workspace
  project: your_project
  version: 2
  license: CC BY 4.0
```

## Label Studio 导入

### 1. 安装 Label Studio

```bash
# 使用 pip 安装
pip install label-studio

# 启动服务
label-studio start
```

访问 http://localhost:8080

### 2. 创建项目

1. 点击 "Create Project"
2. 输入项目名称（如 "YOLO Dataset Review"）
3. 点击 "Save"

### 3. 配置标注界面

在 "Labeling Setup" 中：

1. 选择 "Custom Template"
2. 复制以下配置（或使用项目中的 `label_studio_config.xml`）：

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2">
    <Label value="body" background="#FF0000" hotkey="1"/>
    <Label value="head" background="#00FF00" hotkey="2"/>
    <Label value="leg" background="#0000FF" hotkey="3"/>
  </RectangleLabels>
</View>
```

3. 点击 "Save"

### 4. 配置存储（重要）

要让 Label Studio 能够读取图像文件：

**方法一：配置本地文件存储**

在 Label Studio 的设置中配置本地文件存储：

1. 进入项目设置 → Storage
2. 点击 "Add Source Storage"
3. 选择 "Local files"
4. 设置路径为您的 `datasets` 文件夹的绝对路径
5. 保存

**方法二：复制文件到 Label Studio**

```bash
# 将 datasets 文件夹复制到 Label Studio 的 media 目录
cp -r datasets /path/to/label-studio/media/
```

### 5. 导入数据

1. 点击 "Import" 按钮
2. 选择生成的 JSON 文件（如 `test_review.json`）
3. 点击 "Import"
4. 等待导入完成

### 6. 开始审核

导入成功后：

1. 点击任意任务开始审核
2. 查看现有标注
3. 修改错误的标注
4. 添加遗漏的标注
5. 删除错误的标注
6. 点击 "Submit" 保存

### 快捷键

- `1` - 选择 body 类别（红色）
- `2` - 选择 head 类别（绿色）
- `3` - 选择 leg 类别（蓝色）
- `Delete` / `Backspace` - 删除选中的标注
- `Ctrl+Z` - 撤销
- `Ctrl+Shift+Z` - 重做
- `空格` - 切换到下一个任务

## 使用示例

### 示例1：转换测试集用于审核

```bash
python yolo2label_studio.py --dataset test --output review_test.json
```

然后在Label Studio中导入 `review_test.json` 进行审核。

### 示例2：转换特定目录的数据

假设您有一个特殊的数据集在 `datasets/special_cases/`：

```bash
python yolo2label_studio.py \
    --dataset-path ./datasets/special_cases \
    --output special_cases_ls.json
```

### 示例3：批量转换所有数据集

```bash
# 创建输出目录
mkdir -p label_studio_imports

# 转换所有数据集
python yolo2label_studio.py --dataset train --output label_studio_imports/train.json
python yolo2label_studio.py --dataset valid --output label_studio_imports/valid.json
python yolo2label_studio.py --dataset test --output label_studio_imports/test.json
```

## 常见问题

### Q1: Label Studio 无法显示图像

**问题**: 导入后图像无法加载，显示错误或空白

**原因**: Label Studio 找不到图像文件路径

**解决方案**:

1. **配置本地存储**（推荐）
   ```bash
   # 在 Label Studio 项目设置中
   # Storage → Add Source Storage → Local files
   # 路径设置为：/path/to/your/datasets
   ```

2. **使用绝对路径**
   编辑生成的JSON文件，将路径改为绝对路径：
   ```json
   "image": "/data/local-files/?d=/absolute/path/to/datasets/test/images/file.jpg"
   ```

3. **复制文件**
   ```bash
   # 将图像复制到 Label Studio 可访问的位置
   cp -r datasets /path/to/label-studio/media/
   ```

### Q2: 类别名称不匹配

**问题**: 显示的类别与预期不符

**原因**: `data.yaml` 中的类别配置与实际不匹配

**解决方案**:

1. 检查 `data.yaml` 文件：
   ```yaml
   nc: 3
   names: ['body', 'head', 'leg']  # 确保顺序正确
   ```

2. 更新 Label Studio 配置，确保类别一致：
   ```xml
   <Label value="body" background="#FF0000" hotkey="1"/>
   <Label value="head" background="#00FF00" hotkey="2"/>
   <Label value="leg" background="#0000FF" hotkey="3"/>
   ```

### Q3: 坐标位置不准确

**问题**: 边界框位置与实际目标不匹配

**原因**: 
- YOLO 标注文件格式错误
- 图像尺寸读取错误

**解决方案**:

1. 检查 YOLO 标注格式（应该是归一化坐标 0-1）：
   ```
   class_id x_center y_center width height
   0 0.5046875 0.4984375 0.0765625 0.1328125
   ```

2. 确认图像文件完整且未损坏

3. 重新运行转换脚本

### Q4: 转换速度慢

**问题**: 大数据集转换耗时较长

**优化建议**:

1. **分批处理**
   ```bash
   # 先转换一小部分测试
   python3 yolo2label_studio.py --dataset test --output test.json
   ```

2. **使用 SSD**
   将数据集放在 SSD 上可以加快读取速度

3. **减少不必要的转换**
   只转换需要审核的数据集

### Q5: 内存不足

**问题**: 转换大型数据集时内存溢出

**解决方案**:

1. 分批次转换（修改脚本以支持范围）
2. 关闭其他应用程序
3. 增加系统虚拟内存

### Q6: 文件名编码问题

**问题**: 文件名包含特殊字符导致错误

**解决方案**:

1. 重命名文件，使用简单的ASCII字符
2. 确保文件系统支持UTF-8编码

### Q7: 权限错误

**问题**: 无法读取文件或写入输出

**解决方案**:

```bash
# 检查文件权限
ls -la datasets/test/images/

# 修改权限（如果需要）
chmod -R 755 datasets/

# 确保输出目录可写
mkdir -p output/
chmod 755 output/
```

## 完整工作流程

### 阶段1: 数据准备

```bash
# 1. 确认数据集结构
ls -R datasets/

# 2. 检查配置文件
cat datasets/data.yaml

# 3. 安装依赖
pip install pillow pyyaml label-studio
```

### 阶段2: 转换数据

```bash
# 4. 转换测试集（建议先从测试集开始）
python3 yolo2label_studio.py --dataset test --output test_review.json

# 5. 检查输出文件
head -100 test_review.json
```

### 阶段3: Label Studio 设置

```bash
# 6. 启动 Label Studio
label-studio start

# 7. 在浏览器中打开
# http://localhost:8080
```

在 Label Studio 中：
1. 创建新项目
2. 配置标注界面（使用 `label_studio_config.xml`）
3. 配置存储（指向 datasets 文件夹）
4. 导入 JSON 文件

### 阶段4: 审核标注

1. 逐个查看图像和标注
2. 修正错误的标注
3. 添加遗漏的目标
4. 删除误标的框
5. 保存修改

### 阶段5: 导出数据

在 Label Studio 中：
1. 点击 "Export" 按钮
2. 选择 "JSON" 格式
3. 下载导出文件

### 阶段6: 转换回 YOLO 格式

```bash
# 将 Label Studio 导出的 JSON 转回 YOLO 格式
# (需要另外的转换脚本)
python3 labelstudio2yolo.py --input exported.json --output datasets/train_corrected/
```

### 阶段7: 重新训练

```bash
# 使用修正后的数据重新训练
python3 train_yolo.py --data datasets/data_corrected.yaml
```

## 最佳实践

### 1. 分批审核

不要一次性审核所有数据，建议：
- 先审核测试集（数量较少）
- 根据测试集发现的问题，针对性审核训练集
- 重点审核混淆矩阵中错误率高的类别

### 2. 制定审核标准

在开始审核前，明确：
- 边界框应该紧贴目标还是留有余量
- 部分遮挡的目标是否标注
- 模糊不清的目标如何处理
- 截断的目标（图像边缘）如何处理

### 3. 团队协作

如果是团队项目：
- 使用 Label Studio 的用户管理功能
- 分配不同的数据集给不同的审核员
- 定期讨论和统一标注标准

### 4. 版本管理

```bash
# 保留原始数据
cp -r datasets datasets_backup

# 为修正后的数据创建新版本
mkdir datasets_v2

# 使用 git 管理配置和脚本
git add *.py *.yaml *.xml
git commit -m "Updated annotation review tools"
```

### 5. 质量检查

审核完成后：
- 随机抽查 10-20% 的数据
- 计算标注一致性
- 与其他审核员交叉验证

## 性能提示

- **大数据集**: 每次转换 100-500 张图像
- **进度跟踪**: 脚本会每 100 张显示进度
- **批量操作**: 使用 `convert_to_labelstudio.sh` 脚本
- **存储空间**: 确保有足够的磁盘空间（JSON 文件可能较大）

## 故障排查清单

运行前检查：

- [ ] Python 版本 >= 3.6
- [ ] 已安装 Pillow 和 PyYAML
- [ ] datasets 目录结构正确
- [ ] data.yaml 文件存在且格式正确
- [ ] 图像和标注文件存在
- [ ] 图像和标注文件名匹配（相同的 base name）
- [ ] 有足够的磁盘空间
- [ ] 输出目录可写

运行后检查：

- [ ] JSON 文件生成成功
- [ ] 文件大小合理（不为空）
- [ ] 可以在文本编辑器中打开
- [ ] JSON 格式正确（可以使用 `jq` 验证）
- [ ] 图像路径格式正确
- [ ] 标注数量与预期一致

## 技术细节

### YOLO 格式说明

```
class_id x_center y_center width height
```

- 所有坐标都是归一化的（0-1范围）
- `x_center`, `y_center`: 边界框中心点
- `width`, `height`: 边界框宽高
- `class_id`: 类别索引（从0开始）

### Label Studio 格式说明

```json
{
  "x": 左上角X坐标（百分比 0-100）,
  "y": 左上角Y坐标（百分比 0-100）,
  "width": 宽度（百分比 0-100）,
  "height": 高度（百分比 0-100）
}
```

### 坐标转换公式

```python
# YOLO -> Label Studio
x = (x_center - width/2) * 100
y = (y_center - height/2) * 100
w = width * 100
h = height * 100
```

## 相关文档

- [Label Studio 快速开始](LABELSTUDIO_QUICKSTART.md)
- [项目结构说明](PROJECT_STRUCTURE.md)
- [训练指南](UNIFIED_TRAIN_GUIDE.md)

## 贡献与反馈

如果您发现任何问题或有改进建议，欢迎：
- 提交 Issue
- 发起 Pull Request
- 联系项目维护者

---

**祝您审核顺利！** 🎉
