#!/bin/bash

# YOLO12 模型导出示例脚本

# 设置模型路径
MODEL_PATH="runs/detect/yolo12n_person_head/weights/best.pt"

echo "🚀 YOLO12 模型导出示例"
echo "======================================"

# 示例1: 导出 ONNX (FP32)
echo ""
echo "示例1: 导出 ONNX (FP32)"
echo "命令: python export_model.py --model $MODEL_PATH --format onnx"
# python export_model.py --model $MODEL_PATH --format onnx

# 示例2: 导出 ONNX (INT8)
echo ""
echo "示例2: 导出 ONNX (INT8 量化)"
echo "命令: python export_model.py --model $MODEL_PATH --format onnx --int8"
# python export_model.py --model $MODEL_PATH --format onnx --int8

# 示例3: 导出 TensorRT (FP16)
echo ""
echo "示例3: 导出 TensorRT (FP16)"
echo "命令: python export_model.py --model $MODEL_PATH --format engine --half"
# python export_model.py --model $MODEL_PATH --format engine --half

# 示例4: 导出 TensorRT (INT8)
echo ""
echo "示例4: 导出 TensorRT (INT8 量化)"
echo "命令: python export_model.py --model $MODEL_PATH --format engine --int8"
# python export_model.py --model $MODEL_PATH --format engine --int8

# 示例5: 同时导出多种格式
echo ""
echo "示例5: 同时导出 ONNX 和 TensorRT (INT8)"
echo "命令: python export_model.py --model $MODEL_PATH --format onnx,engine --int8"
# python export_model.py --model $MODEL_PATH --format onnx,engine --int8

echo ""
echo "======================================"
echo "💡 取消注释相应的命令行即可运行"
echo "📁 导出的模型将保存在模型同目录下"
