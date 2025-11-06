"""
YOLO12 目标检测训练脚本 - 云服务器版本
适用于 NVIDIA T4 GPU (16GB)
CPU: 8核, 内存: 16GB

主要优化:
- 使用 CUDA 加速
- 增加 batch size
- 增加 workers 数量
- 优化训练速度
"""

from ultralytics import YOLO
import torch
import os

def main():
    # ============ 配置参数 (T4 GPU 优化) ============
    
    # 数据集配置文件路径
    data_yaml = 'datasets/data.yaml'
    
    # 模型选择
    # 选项: yolo12n.pt (最快), yolo12s.pt (推荐), yolo12m.pt (更好效果)
    model_name = 'yolo12s.pt'  # T4可以用更大的模型
    
    # 训练参数 - T4 GPU优化
    epochs = 100             # T4速度快，可以训练更多轮
    batch_size = 32          # T4 16GB可以用32，如果显存不足降到24或16
    img_size = 640           # 图像尺寸
    
    # 设备设置
    device = 0               # 使用第一块GPU (cuda:0)
    
    # 数据加载优化 - 8核CPU
    workers = 8              # 设置为CPU核心数
    
    # 训练策略
    patience = 30            # 早停耐心值，可以更大因为训练快
    cache = 'ram'            # 将数据集缓存到内存(16GB够用)，加速训练
    
    # 保存路径
    project_name = 'runs/detect'
    experiment_name = 'yolo12s_person_head_t4'  # 标注是T4训练的
    
    # ============ 检查环境 ============
    print("=" * 60)
    print("🚀 YOLO12 训练配置 (云服务器 - T4 GPU)")
    print("=" * 60)
    print(f"📊 数据集: {data_yaml}")
    print(f"🤖 模型: {model_name}")
    print(f"💻 设备: CUDA (GPU)")
    print(f"🔢 Epochs: {epochs}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"📐 图像尺寸: {img_size}")
    print(f"👷 Workers: {workers}")
    print(f"💾 缓存: {cache}")
    print("=" * 60)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("\n⚠️  警告: CUDA不可用，将使用CPU训练（会很慢）")
        device = 'cpu'
        batch_size = 8
        workers = 4
        cache = False
    else:
        print(f"\n✅ CUDA可用")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查数据集是否存在
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # ============ 加载模型 ============
    print("\n📥 加载模型...")
    model = YOLO(model_name)
    print(f"✅ 模型加载成功: {model_name}")
    
    # ============ 开始训练 ============
    print("\n🏋️ 开始训练...\n")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            workers=workers,
            cache=cache,              # 缓存数据集到内存
            
            # 优化参数
            patience=patience,
            save=True,
            save_period=20,           # 每20个epoch保存一次（训练快了可以少保存）
            
            # 优化器设置 (T4可以用更激进的设置)
            optimizer='AdamW',        # AdamW通常比SGD收敛快
            lr0=0.001,               # 初始学习率
            lrf=0.01,                # 最终学习率
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,        # warmup轮数
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # 数据增强 (可以更激进)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,              # T4可以用mixup增强
            copy_paste=0.0,
            
            # 保存设置
            project=project_name,
            name=experiment_name,
            exist_ok=True,
            
            # 其他
            pretrained=True,
            verbose=True,
            seed=42,
            deterministic=False,      # T4上可以关闭确定性以获得更快速度
            single_cls=False,         # 多类别检测
            rect=False,               # 矩形训练（可选）
            cos_lr=True,              # 使用余弦学习率衰减
            close_mosaic=10,          # 最后10个epoch关闭mosaic
            
            # 验证设置
            val=True,
            plots=True,
            
            # GPU优化
            amp=True,                 # 自动混合精度，加速训练并节省显存
        )
        
        print("\n" + "=" * 60)
        print("🎉 训练完成!")
        print("=" * 60)
        
        # 显示最佳模型路径
        best_model_path = f"{project_name}/{experiment_name}/weights/best.pt"
        last_model_path = f"{project_name}/{experiment_name}/weights/last.pt"
        
        print(f"\n📁 训练结果保存在: {project_name}/{experiment_name}/")
        print(f"🏆 最佳模型: {best_model_path}")
        print(f"📝 最后模型: {last_model_path}")
        
        # ============ 验证模型 ============
        print("\n🔍 在验证集上评估最佳模型...")
        
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=data_yaml, device=device)
        
        print("\n📊 验证集性能指标:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        # ============ 在测试集上评估 ============
        print("\n🔍 在测试集上评估最佳模型...")
        test_metrics = best_model.val(data=data_yaml, split='test', device=device)
        
        print("\n📊 测试集性能指标:")
        print(f"  mAP50: {test_metrics.box.map50:.4f}")
        print(f"  mAP50-95: {test_metrics.box.map:.4f}")
        print(f"  Precision: {test_metrics.box.mp:.4f}")
        print(f"  Recall: {test_metrics.box.mr:.4f}")
        
        print("\n✅ 所有任务完成!")
        
        # 显示训练时间统计
        print("\n⏱️  训练统计:")
        print(f"  预计训练时间: ~{epochs * 0.5:.0f}-{epochs * 1:.0f}分钟 (T4)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print(f"💾 部分训练结果已保存在: {project_name}/{experiment_name}/")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    # 打印系统信息
    print("\n🖥️  系统信息:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 开始训练
    main()
