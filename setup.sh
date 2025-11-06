#!/bin/bash

# YOLO12 训练环境设置脚本
# 适用于 MacBook Air M1

echo "========================================"
echo "🚀 YOLO12 训练环境设置"
echo "========================================"

# 检查Python版本
echo ""
echo "📌 检查Python版本..."
python3 --version

# 检查是否存在虚拟环境
if [ -d "venv" ]; then
    echo "✅ 虚拟环境已存在"
else
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo ""
echo "🔌 激活虚拟环境..."
source venv/bin/activate

# 更新pip
echo ""
echo "⬆️  更新pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "📥 安装依赖包..."
pip install -r requirements.txt

# 检查安装
echo ""
echo "========================================"
echo "🔍 检查关键依赖安装情况"
echo "========================================"

python3 << EOF
import sys

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False

print("")
check_package("PyTorch", "torch")
check_package("Torchvision", "torchvision")
check_package("Ultralytics", "ultralytics")
check_package("OpenCV", "cv2")
check_package("NumPy", "numpy")
check_package("Matplotlib", "matplotlib")

# 检查MPS支持
print("")
print("🖥️  Apple Silicon (MPS) 支持检查:")
import torch
print(f"  - MPS可用: {torch.backends.mps.is_available()}")
print(f"  - MPS已构建: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("  ✅ 可以使用MPS加速训练!")
else:
    print("  ⚠️  MPS不可用，将使用CPU训练")
EOF

echo ""
echo "========================================"
echo "✅ 环境设置完成!"
echo "========================================"
echo ""
echo "📝 下一步操作："
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 开始训练: python train_yolo.py"
echo "  3. 测试模型: python test_yolo.py"
echo ""
echo "💡 提示: 如需停用虚拟环境，运行 'deactivate'"
echo ""
