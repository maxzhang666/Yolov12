"""
YOLO12 统一训练脚本
支持 M1 和 T4 GPU，通过配置文件或命令行参数选择

使用方式：
  # 方式1: 使用配置文件（推荐）
  python train.py --config m1_standard        # M1 标准配置
  python train.py --config t4_standard        # T4 标准配置
  python train.py --config m1_quick           # M1 快速测试
  python train.py --config t4_highquality     # T4 高质量
  
  # 方式2: 直接指定参数
  python train.py --device mps --model yolo12n.pt --epochs 50
  python train.py --device cuda --model yolo12s.pt --epochs 100
  
  # 方式3: 使用默认配置（根据设备自动选择）
  python train.py                             # 自动检测设备
"""

from ultralytics import YOLO
import torch
import os
import argparse
from train_config import (
    QuickTestConfig, 
    StandardConfig, 
    HighQualityConfig, 
    M1OptimizedConfig
)


# ============ 预设配置映射 ============
CONFIGS = {
    # M1 配置
    'm1_quick': QuickTestConfig,
    'm1_standard': StandardConfig,
    'm1_optimized': M1OptimizedConfig,
    
    # T4 配置（需要修改的参数）
    't4_quick': {
        'MODEL_NAME': 'yolo12n.pt',
        'EPOCHS': 10,
        'BATCH_SIZE': 48,
        'DEVICE': 'cuda',
        'WORKERS': 8,
        'CACHE': 'ram',
        'EXPERIMENT_NAME': 't4_quick_test',
    },
    't4_standard': {
        'MODEL_NAME': 'yolo12s.pt',
        'EPOCHS': 100,
        'BATCH_SIZE': 32,
        'DEVICE': 'cuda',
        'WORKERS': 8,
        'CACHE': 'ram',
        'OPTIMIZER': 'AdamW',
        'LR0': 0.001,
        'MIXUP': 0.15,
        'PATIENCE': 30,
        'EXPERIMENT_NAME': 't4_standard',
    },
    't4_highquality': {
        'MODEL_NAME': 'yolo12m.pt',
        'EPOCHS': 150,
        'BATCH_SIZE': 24,
        'DEVICE': 'cuda',
        'WORKERS': 8,
        'CACHE': 'ram',
        'OPTIMIZER': 'AdamW',
        'LR0': 0.001,
        'MIXUP': 0.2,
        'PATIENCE': 50,
        'EXPERIMENT_NAME': 't4_highquality',
    },
}


def get_config(config_name=None, args=None):
    """
    获取配置
    
    Args:
        config_name: 配置名称
        args: 命令行参数
    
    Returns:
        配置字典
    """
    # 基础配置（使用 StandardConfig 作为基础）
    base_config = StandardConfig()
    
    config = {
        'DATA_YAML': base_config.DATA_YAML,
        'MODEL_NAME': base_config.MODEL_NAME,
        'EPOCHS': base_config.EPOCHS,
        'BATCH_SIZE': base_config.BATCH_SIZE,
        'IMG_SIZE': base_config.IMG_SIZE,
        'DEVICE': base_config.DEVICE,
        'WORKERS': base_config.WORKERS,
        'PATIENCE': base_config.PATIENCE,
        'PROJECT_NAME': base_config.PROJECT_NAME,
        'EXPERIMENT_NAME': base_config.EXPERIMENT_NAME,
        'SAVE_PERIOD': base_config.SAVE_PERIOD,
        'OPTIMIZER': base_config.OPTIMIZER,
        'LR0': base_config.LR0,
        'LRF': base_config.LRF,
        'MOMENTUM': base_config.MOMENTUM,
        'WEIGHT_DECAY': base_config.WEIGHT_DECAY,
        'HSV_H': base_config.HSV_H,
        'HSV_S': base_config.HSV_S,
        'HSV_V': base_config.HSV_V,
        'DEGREES': base_config.DEGREES,
        'TRANSLATE': base_config.TRANSLATE,
        'SCALE': base_config.SCALE,
        'SHEAR': base_config.SHEAR,
        'PERSPECTIVE': base_config.PERSPECTIVE,
        'FLIPUD': base_config.FLIPUD,
        'FLIPLR': base_config.FLIPLR,
        'MOSAIC': base_config.MOSAIC,
        'MIXUP': base_config.MIXUP,
        'AMP': base_config.AMP,
        'PRETRAINED': base_config.PRETRAINED,
        'VERBOSE': base_config.VERBOSE,
        'SEED': base_config.SEED,
        'DETERMINISTIC': base_config.DETERMINISTIC,
        'PLOTS': base_config.PLOTS,
        'VAL': base_config.VAL,
        'CACHE': False,  # 默认不缓存
    }
    
    # 如果指定了配置名称
    if config_name and config_name in CONFIGS:
        preset = CONFIGS[config_name]
        if isinstance(preset, type):
            # 是配置类
            preset_obj = preset()
            for key in config.keys():
                if hasattr(preset_obj, key):
                    config[key] = getattr(preset_obj, key)
        else:
            # 是字典
            config.update(preset)
    
    # 命令行参数覆盖
    if args:
        if args.model:
            config['MODEL_NAME'] = args.model
        if args.epochs:
            config['EPOCHS'] = args.epochs
        if args.batch:
            config['BATCH_SIZE'] = args.batch
        if args.device:
            config['DEVICE'] = args.device
        if args.workers:
            config['WORKERS'] = args.workers
        if args.cache:
            config['CACHE'] = args.cache
    
    # 自动检测设备
    if config['DEVICE'] == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS 不可用，切换到 CPU")
        config['DEVICE'] = 'cpu'
    elif config['DEVICE'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，切换到 CPU")
        config['DEVICE'] = 'cpu'
    
    return config


def main():
    # ============ 解析命令行参数 ============
    parser = argparse.ArgumentParser(description='YOLO12 训练脚本')
    
    # 配置选择
    parser.add_argument('--config', type=str, 
                        choices=list(CONFIGS.keys()),
                        help='预设配置名称')
    
    # 核心参数
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--device', type=str, 
                        choices=['mps', 'cuda', 'cpu', '0', '1'],
                        help='设备类型')
    parser.add_argument('--workers', type=int, help='数据加载线程数')
    parser.add_argument('--cache', type=str, 
                        choices=['ram', 'disk', 'false'],
                        help='数据缓存方式')
    
    args = parser.parse_args()
    
    # ============ 获取配置 ============
    config = get_config(args.config, args)
    
    # ============ 显示配置信息 ============
    print("\n" + "=" * 70)
    print("🚀 YOLO12 训练配置")
    print("=" * 70)
    
    if args.config:
        print(f"📂 预设配置: {args.config}")
    else:
        print(f"📂 使用默认配置")
    
    print(f"\n💻 设备信息:")
    print(f"  目标设备: {config['DEVICE']}")
    if config['DEVICE'] == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif config['DEVICE'] == 'mps':
        print(f"  Apple Silicon MPS 加速")
    
    print(f"\n📊 训练参数:")
    print(f"  数据集: {config['DATA_YAML']}")
    print(f"  模型: {config['MODEL_NAME']}")
    print(f"  Epochs: {config['EPOCHS']}")
    print(f"  Batch Size: {config['BATCH_SIZE']}")
    print(f"  Image Size: {config['IMG_SIZE']}")
    print(f"  Workers: {config['WORKERS']}")
    print(f"  Optimizer: {config['OPTIMIZER']}")
    print(f"  Learning Rate: {config['LR0']}")
    if config['CACHE']:
        print(f"  Cache: {config['CACHE']} ⚡")
    
    print(f"\n📁 输出路径:")
    print(f"  项目: {config['PROJECT_NAME']}/{config['EXPERIMENT_NAME']}")
    print("=" * 70)
    
    # 确认开始
    if args.config:
        print(f"\n✅ 将使用 '{args.config}' 配置开始训练")
    else:
        print(f"\n✅ 将使用默认配置开始训练")
    
    # ============ 检查数据集 ============
    if not os.path.exists(config['DATA_YAML']):
        raise FileNotFoundError(f"数据集配置文件不存在: {config['DATA_YAML']}")
    
    # ============ 加载模型 ============
    print("\n📥 加载模型...")
    model = YOLO(config['MODEL_NAME'])
    print(f"✅ 模型加载成功: {config['MODEL_NAME']}")
    
    # ============ 开始训练 ============
    print("\n🏋️ 开始训练...\n")
    
    try:
        # 准备训练参数
        train_args = {
            'data': config['DATA_YAML'],
            'epochs': config['EPOCHS'],
            'batch': config['BATCH_SIZE'],
            'imgsz': config['IMG_SIZE'],
            'device': config['DEVICE'],
            'workers': config['WORKERS'],
            'patience': config['PATIENCE'],
            'save': True,
            'save_period': config['SAVE_PERIOD'],
            'optimizer': config['OPTIMIZER'],
            'lr0': config['LR0'],
            'lrf': config['LRF'],
            'momentum': config['MOMENTUM'],
            'weight_decay': config['WEIGHT_DECAY'],
            'hsv_h': config['HSV_H'],
            'hsv_s': config['HSV_S'],
            'hsv_v': config['HSV_V'],
            'degrees': config['DEGREES'],
            'translate': config['TRANSLATE'],
            'scale': config['SCALE'],
            'shear': config['SHEAR'],
            'perspective': config['PERSPECTIVE'],
            'flipud': config['FLIPUD'],
            'fliplr': config['FLIPLR'],
            'mosaic': config['MOSAIC'],
            'mixup': config['MIXUP'],
            'project': config['PROJECT_NAME'],
            'name': config['EXPERIMENT_NAME'],
            'exist_ok': True,
            'pretrained': config['PRETRAINED'],
            'verbose': config['VERBOSE'],
            'seed': config['SEED'],
            'deterministic': config['DETERMINISTIC'],
            'val': config['VAL'],
            'plots': config['PLOTS'],
            'amp': config['AMP'],
        }
        
        # 添加缓存参数（如果启用）
        if config['CACHE']:
            train_args['cache'] = config['CACHE']
        
        # 开始训练
        results = model.train(**train_args)
        
        print("\n" + "=" * 70)
        print("🎉 训练完成!")
        print("=" * 70)
        
        # 显示结果路径
        best_path = f"{config['PROJECT_NAME']}/{config['EXPERIMENT_NAME']}/weights/best.pt"
        print(f"\n🏆 最佳模型: {best_path}")
        
        # 验证模型
        print("\n🔍 在验证集上评估...")
        best_model = YOLO(best_path)
        metrics = best_model.val(data=config['DATA_YAML'], device=config['DEVICE'])
        
        print("\n📊 性能指标:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        print("\n✅ 所有任务完成!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        raise


if __name__ == '__main__':
    # 显示可用配置
    print("\n📋 可用的预设配置:")
    print("\nM1 配置:")
    print("  m1_quick      - 快速测试 (5 epochs)")
    print("  m1_standard   - 标准训练 (50 epochs) ⭐")
    print("  m1_optimized  - M1优化 (内存友好)")
    print("\nT4 配置:")
    print("  t4_quick      - 快速测试 (10 epochs, yolo12n)")
    print("  t4_standard   - 标准训练 (100 epochs, yolo12s) ⭐")
    print("  t4_highquality - 高质量 (150 epochs, yolo12m)")
    print("\n" + "=" * 70)
    
    main()
