#!/bin/bash
# YOLO to Label Studio 转换脚本示例

echo "======================================"
echo "YOLO to Label Studio 批量转换工具"
echo "======================================"
echo ""

# 创建输出目录
OUTPUT_DIR="label_studio_exports"
mkdir -p "$OUTPUT_DIR"

echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import PIL, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install pillow pyyaml
fi

echo ""
echo "======================================"
echo "选择要转换的数据集:"
echo "======================================"
echo "1. 训练集 (train)"
echo "2. 验证集 (valid)"
echo "3. 测试集 (test)"
echo "4. 全部转换"
echo "5. 自定义路径"
echo ""

read -p "请选择 (1-5): " choice

case $choice in
    1)
        echo "转换训练集..."
        python3 yolo2label_studio.py \
            --dataset train \
            --output "$OUTPUT_DIR/train_labelstudio.json"
        ;;
    2)
        echo "转换验证集..."
        python3 yolo2label_studio.py \
            --dataset valid \
            --output "$OUTPUT_DIR/valid_labelstudio.json"
        ;;
    3)
        echo "转换测试集..."
        python3 yolo2label_studio.py \
            --dataset test \
            --output "$OUTPUT_DIR/test_labelstudio.json"
        ;;
    4)
        echo "转换所有数据集..."
        
        echo "1/3 转换训练集..."
        python3 yolo2label_studio.py \
            --dataset train \
            --output "$OUTPUT_DIR/train_labelstudio.json"
        
        echo ""
        echo "2/3 转换验证集..."
        python3 yolo2label_studio.py \
            --dataset valid \
            --output "$OUTPUT_DIR/valid_labelstudio.json"
        
        echo ""
        echo "3/3 转换测试集..."
        python3 yolo2label_studio.py \
            --dataset test \
            --output "$OUTPUT_DIR/test_labelstudio.json"
        ;;
    5)
        read -p "请输入数据集路径: " custom_path
        read -p "请输入输出文件名: " output_name
        
        echo "转换自定义数据集..."
        python3 yolo2label_studio.py \
            --dataset-path "$custom_path" \
            --output "$OUTPUT_DIR/$output_name"
        ;;
    *)
        echo "无效的选择"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "转换完成!"
echo "======================================"
echo ""
echo "生成的文件位于: $OUTPUT_DIR/"
echo ""
echo "下一步操作:"
echo "1. 打开 Label Studio"
echo "2. 创建新项目或打开现有项目"
echo "3. 导入生成的 JSON 文件"
echo "4. 开始审核和标注"
echo ""
echo "详细使用指南请查看: docs/YOLO_TO_LABELSTUDIO.md"
echo ""
