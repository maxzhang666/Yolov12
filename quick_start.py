"""
YOLO12 快速开始示例
这个脚本展示了最简单的训练和预测流程
"""

from ultralytics import YOLO
import torch


def quick_start():
    """快速开始 - 5分钟完成整个流程"""
    
    print("=" * 60)
    print("🚀 YOLO12 快速开始")
    print("=" * 60)
    
    # 步骤 1: 检查环境
    print("\n📋 步骤 1/4: 检查环境")
    print(f"  Python: ✅")
    print(f"  PyTorch: {torch.__version__} ✅")
    print(f"  MPS可用: {'✅' if torch.backends.mps.is_available() else '❌'}")
    
    # 步骤 2: 快速训练（5个epoch用于测试）
    print("\n🏋️  步骤 2/4: 快速训练（5 epochs，约5分钟）")
    print("  提示: 这只是快速测试，完整训练请运行 train_yolo.py")
    
    model = YOLO('yolo12n.pt')  # 加载预训练模型
    
    results = model.train(
        data='datasets/data.yaml',
        epochs=5,                # 快速测试只训练5轮
        batch=8,
        imgsz=640,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        project='runs/detect',
        name='quick_test',
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    print("\n✅ 快速训练完成!")
    
    # 步骤 3: 验证模型
    print("\n🔍 步骤 3/4: 验证模型")
    
    model = YOLO('runs/detect/quick_test/weights/best.pt')
    metrics = model.val(data='datasets/data.yaml')
    
    print(f"\n📊 性能指标:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  精确率:   {metrics.box.mp:.4f}")
    print(f"  召回率:   {metrics.box.mr:.4f}")
    
    # 步骤 4: 测试预测
    print("\n🎯 步骤 4/4: 测试预测")
    
    test_images = 'datasets/test/images'
    results = model.predict(
        source=test_images,
        conf=0.25,
        save=True,
        project='runs/predict',
        name='quick_test',
        exist_ok=True
    )
    
    print(f"\n✅ 预测完成!")
    print(f"  结果保存在: runs/predict/quick_test/")
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 快速开始完成!")
    print("=" * 60)
    print("\n📝 下一步:")
    print("  1. 查看训练曲线: runs/detect/quick_test/results.png")
    print("  2. 查看预测结果: runs/predict/quick_test/")
    print("  3. 完整训练: python train_yolo.py")
    print("\n💡 提示:")
    print("  - 快速测试只训练了5轮，效果可能不佳")
    print("  - 完整训练建议50-100轮")
    print("  - 使用 python train_yolo.py 进行正式训练")
    print("=" * 60)


if __name__ == '__main__':
    try:
        quick_start()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("\n💡 解决方案:")
        print("  1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("  2. 确保数据集在 datasets/ 目录下")
        print("  3. 查看 GUIDE.md 获取详细帮助")
