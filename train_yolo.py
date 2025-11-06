"""
YOLO12 目标检测训练脚本
适用于 MacBook Air M1
检测目标: person, head

使用方式：
  方式1（推荐新手）: 直接运行，使用下面的默认配置
    python train_yolo.py
  
  方式2（使用配置类）: 从 train_config.py 导入配置
    # 取消下面的注释，使用预设配置
    # from train_config import StandardConfig as Config
"""

from ultralytics import YOLO
import torch
import os

# ============ 配置方式选择 ============
# 取消下面某一行的注释，使用对应的配置类
# from train_config import QuickTestConfig as Config
# from train_config import StandardConfig as Config
# from train_config import HighQualityConfig as Config
# from train_config import M1OptimizedConfig as Config

# 如果上面没有导入Config，则使用下面的默认配置
try:
    Config
    USE_CONFIG_FILE = True
    print("✅ 使用配置文件: train_config.py")
except NameError:
    USE_CONFIG_FILE = False
    print("✅ 使用默认配置（脚本内置）")


def main():
    # ============ 配置参数 ============
    
    if USE_CONFIG_FILE:
        # 从配置文件读取
        data_yaml = Config.DATA_YAML
        model_name = Config.MODEL_NAME
        epochs = Config.EPOCHS
        batch_size = Config.BATCH_SIZE
        img_size = Config.IMG_SIZE
        device = Config.DEVICE if Config.DEVICE != 'mps' else ('mps' if torch.backends.mps.is_available() else 'cpu')
        workers = Config.WORKERS
        patience = Config.PATIENCE
        project_name = Config.PROJECT_NAME
        experiment_name = Config.EXPERIMENT_NAME
        save_period = Config.SAVE_PERIOD
        optimizer = Config.OPTIMIZER
        lr0 = Config.LR0
        lrf = Config.LRF
        momentum = Config.MOMENTUM
        weight_decay = Config.WEIGHT_DECAY
        hsv_h = Config.HSV_H
        hsv_s = Config.HSV_S
        hsv_v = Config.HSV_V
        degrees = Config.DEGREES
        translate = Config.TRANSLATE
        scale = Config.SCALE
        shear = Config.SHEAR
        perspective = Config.PERSPECTIVE
        flipud = Config.FLIPUD
        fliplr = Config.FLIPLR
        mosaic = Config.MOSAIC
        mixup = Config.MIXUP
        amp = Config.AMP
        pretrained = Config.PRETRAINED
        verbose = Config.VERBOSE
        seed = Config.SEED
        deterministic = Config.DETERMINISTIC
        plots = Config.PLOTS
        val = Config.VAL
    else:
        # 使用默认配置
        # 数据集配置文件路径
        data_yaml = 'datasets/data.yaml'
        
        # 模型选择 (yolov12n - nano 版本，最轻量)
        model_name = 'yolo12n.pt'
        
        # 训练参数
        epochs = 50              # 训练轮数，可根据效果调整到100
        batch_size = 16          # 批次大小，M1建议8-16，如果内存不足可降低
        img_size = 640           # 图像尺寸
        
        # 设备设置 (M1自动使用MPS加速)
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # 其他训练参数
        workers = 4              # 数据加载的线程数，M1建议4-8
        patience = 20            # 早停耐心值，20个epoch无改善则停止
        
        # 保存路径
        project_name = 'runs/detect'
        experiment_name = 'yolo12n_person_head'
        save_period = 10
        
        # 优化器参数
        optimizer = 'auto'
        lr0 = 0.01
        lrf = 0.01
        momentum = 0.937
        weight_decay = 0.0005
        
        # 数据增强
        hsv_h = 0.015
        hsv_s = 0.7
        hsv_v = 0.4
        degrees = 0.0
        translate = 0.1
        scale = 0.5
        shear = 0.0
        perspective = 0.0
        flipud = 0.0
        fliplr = 0.5
        mosaic = 1.0
        mixup = 0.0
        
        # 其他
        amp = True
        pretrained = True
        verbose = True
        seed = 42
        deterministic = True
        plots = True
        val = True
    
    # ============ 检查环境 ============
    print("=" * 60)
    print(f"🚀 YOLO12 训练配置 {'(使用配置文件)' if USE_CONFIG_FILE else '(默认配置)'}")
    print("=" * 60)
    print(f"📊 数据集: {data_yaml}")
    print(f"🤖 模型: {model_name}")
    print(f"💻 设备: {device}")
    print(f"🔢 Epochs: {epochs}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"📐 图像尺寸: {img_size}")
    print(f"👷 Workers: {workers}")
    print(f"⚡ Optimizer: {optimizer}")
    if USE_CONFIG_FILE:
        print(f"📂 配置来源: train_config.py -> {Config.__name__}")
    print("=" * 60)
    
    # 检查数据集是否存在
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # ============ 加载模型 ============
    print("\n📥 加载模型...")
    
    # 如果是第一次运行，会自动下载预训练权重
    model = YOLO(model_name)
    
    print(f"✅ 模型加载成功: {model_name}")
    
    # ============ 开始训练 ============
    print("\n🏋️ 开始训练...\n")
    
    try:
        results = model.train(
            data=data_yaml,           # 数据集配置文件
            epochs=epochs,            # 训练轮数
            batch=batch_size,         # 批次大小
            imgsz=img_size,           # 图像尺寸
            device=device,            # 使用MPS加速
            workers=workers,          # 数据加载线程
            
            # 优化参数
            patience=patience,        # 早停耐心值
            save=True,                # 保存检查点
            save_period=save_period,  # 保存周期
            
            # 优化器设置
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            
            # 数据增强（可根据需要调整）
            hsv_h=hsv_h,             # 色调增强
            hsv_s=hsv_s,             # 饱和度增强
            hsv_v=hsv_v,             # 亮度增强
            degrees=degrees,         # 旋转角度
            translate=translate,     # 平移
            scale=scale,             # 缩放
            shear=shear,             # 剪切
            perspective=perspective, # 透视变换
            flipud=flipud,           # 上下翻转
            fliplr=fliplr,           # 左右翻转
            mosaic=mosaic,           # Mosaic增强
            mixup=mixup,             # Mixup增强
            
            # 保存设置
            project=project_name,     # 项目文件夹
            name=experiment_name,     # 实验名称
            exist_ok=True,           # 如果文件夹存在则覆盖
            
            # 其他
            pretrained=pretrained,   # 使用预训练权重
            verbose=verbose,         # 详细输出
            seed=seed,               # 随机种子
            deterministic=deterministic,  # 确定性训练
            
            # 验证设置
            val=val,                 # 每个epoch后进行验证
            plots=plots,             # 生成训练图表
            
            # M1优化
            amp=amp,                 # 自动混合精度训练（加速）
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
        
        print("\n✅ 所有任务完成!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print(f"💾 部分训练结果已保存在: {project_name}/{experiment_name}/")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    # 打印系统信息
    print("\n🖥️  系统信息:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")
    print(f"MPS已构建: {torch.backends.mps.is_built()}")
    
    # 开始训练
    main()
