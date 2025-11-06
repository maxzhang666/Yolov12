"""
配置对比 - M1 vs T4 GPU
快速参考指南
"""


class ConfigComparison:
    """配置参数对比"""
    
    # ============ MacBook Air M1 配置 ============
    class M1Config:
        """M1 本地训练配置"""
        
        # 基础参数
        MODEL = 'yolo12n.pt'              # 轻量模型
        EPOCHS = 50                       # 50轮
        BATCH_SIZE = 16                   # 较小批次
        IMG_SIZE = 640
        
        # 设备
        DEVICE = 'mps'                    # Apple Silicon
        
        # 性能
        WORKERS = 4                       # 4核够用
        CACHE = False                     # 不缓存（内存紧张）
        
        # 优化器
        OPTIMIZER = 'auto'
        LR0 = 0.01
        
        # 增强
        MIXUP = 0.0                       # 不用mixup
        
        # 其他
        PATIENCE = 20
        SAVE_PERIOD = 10
        DETERMINISTIC = True              # 确保可复现
        
        # 预期
        TRAINING_TIME = "30-60分钟"
        GPU_MEMORY = "4-8GB (共享内存)"
        EXPECTED_MAP = "0.50-0.60"
    
    
    # ============ T4 GPU 云服务器配置 ============
    class T4Config:
        """T4 GPU 云服务器配置"""
        
        # 基础参数
        MODEL = 'yolo12s.pt'              # 更大模型 ⬆️
        EPOCHS = 100                      # 更多轮 ⬆️
        BATCH_SIZE = 32                   # 更大批次 ⬆️
        IMG_SIZE = 640
        
        # 设备
        DEVICE = 0                        # CUDA GPU
        
        # 性能
        WORKERS = 8                       # 8核CPU ⬆️
        CACHE = 'ram'                     # 缓存到内存 ⚡
        
        # 优化器
        OPTIMIZER = 'AdamW'               # 更快收敛 ⚡
        LR0 = 0.001                       # 适配AdamW
        COS_LR = True                     # 余弦学习率 ⚡
        
        # 增强
        MIXUP = 0.15                      # 启用mixup ⬆️
        
        # 其他
        PATIENCE = 30                     # 更大耐心 ⬆️
        SAVE_PERIOD = 20                  # 减少保存频率
        DETERMINISTIC = False             # 速度优先 ⚡
        
        # 预期
        TRAINING_TIME = "30-50分钟"        # 快3-5倍 ⚡
        GPU_MEMORY = "8-10GB (独立显存)"
        EXPECTED_MAP = "0.55-0.65"         # 效果更好 ⬆️


# ============ 性能对比表 ============

PERFORMANCE_COMPARISON = {
    'M1 (yolo12n, 50 epochs)': {
        'training_time': '30-60分钟',
        'model_size': '~6MB',
        'inference_speed': '30-60 FPS',
        'expected_map': '0.50-0.60',
        'cost': '免费',
        'pros': ['方便', '成本低', '够用'],
        'cons': ['速度慢', '模型小', '效果一般']
    },
    
    'T4 (yolo12s, 100 epochs)': {
        'training_time': '30-50分钟',
        'model_size': '~12MB',
        'inference_speed': '~2ms/img (T4)',
        'expected_map': '0.55-0.65',
        'cost': '付费（约$0.5-1/小时）',
        'pros': ['快3-5倍', '模型更大', '效果更好', '可训练更久'],
        'cons': ['需要上传数据', '付费', '需要网络']
    },
    
    'T4 (yolo12m, 150 epochs)': {
        'training_time': '60-90分钟',
        'model_size': '~26MB',
        'inference_speed': '~5ms/img (T4)',
        'expected_map': '0.60-0.70',
        'cost': '付费（约$1-2/小时）',
        'pros': ['效果最佳', '模型最大', '精度最高'],
        'cons': ['训练时间长', '成本高', '模型大']
    }
}


# ============ 调整建议 ============

ADJUSTMENT_GUIDE = {
    '从 M1 迁移到 T4': {
        '必须改': [
            'device: mps → 0 (或 cuda)',
        ],
        '建议改': [
            'model: yolo12n.pt → yolo12s.pt',
            'batch_size: 16 → 32',
            'epochs: 50 → 100',
            'workers: 4 → 8',
        ],
        '可选改': [
            'cache: False → ram',
            'optimizer: auto → AdamW',
            'mixup: 0.0 → 0.15',
            'patience: 20 → 30',
            'deterministic: True → False',
        ]
    },
    
    '显存不足时': {
        '降低': [
            'batch_size: 32 → 24 → 16',
            'workers: 8 → 4',
            'cache: ram → False',
        ],
        '保持': [
            'model, epochs, img_size'
        ]
    },
    
    '追求速度': {
        '优化': [
            'cache = ram',
            'amp = True',
            'workers = 8',
            'batch_size = 尽可能大',
            'deterministic = False',
        ]
    },
    
    '追求效果': {
        '优化': [
            'model = yolo12m.pt',
            'epochs = 150',
            'patience = 50',
            'mixup = 0.2',
            '更多数据增强',
        ]
    }
}


# ============ 使用示例 ============

def print_comparison():
    """打印配置对比"""
    print("=" * 80)
    print("📊 M1 vs T4 配置对比")
    print("=" * 80)
    
    print("\n【MacBook Air M1】")
    print(f"  模型: {ConfigComparison.M1Config.MODEL}")
    print(f"  Epochs: {ConfigComparison.M1Config.EPOCHS}")
    print(f"  Batch Size: {ConfigComparison.M1Config.BATCH_SIZE}")
    print(f"  Device: {ConfigComparison.M1Config.DEVICE}")
    print(f"  Workers: {ConfigComparison.M1Config.WORKERS}")
    print(f"  Cache: {ConfigComparison.M1Config.CACHE}")
    print(f"  预计时间: {ConfigComparison.M1Config.TRAINING_TIME}")
    print(f"  预期mAP: {ConfigComparison.M1Config.EXPECTED_MAP}")
    
    print("\n【T4 GPU 云服务器】⭐")
    print(f"  模型: {ConfigComparison.T4Config.MODEL} ⬆️")
    print(f"  Epochs: {ConfigComparison.T4Config.EPOCHS} ⬆️")
    print(f"  Batch Size: {ConfigComparison.T4Config.BATCH_SIZE} ⬆️")
    print(f"  Device: {ConfigComparison.T4Config.DEVICE}")
    print(f"  Workers: {ConfigComparison.T4Config.WORKERS} ⬆️")
    print(f"  Cache: {ConfigComparison.T4Config.CACHE} ⚡")
    print(f"  Optimizer: {ConfigComparison.T4Config.OPTIMIZER} ⚡")
    print(f"  预计时间: {ConfigComparison.T4Config.TRAINING_TIME} ⚡")
    print(f"  预期mAP: {ConfigComparison.T4Config.EXPECTED_MAP} ⬆️")
    
    print("\n" + "=" * 80)
    print("💡 总结:")
    print("  - T4 速度快 3-5倍")
    print("  - T4 可用更大模型、更多训练轮数")
    print("  - T4 效果预期提升 5-10%")
    print("  - M1 适合测试和原型，T4 适合正式训练")
    print("=" * 80)


if __name__ == '__main__':
    print_comparison()
    
    print("\n\n📋 详细性能对比:\n")
    for config_name, details in PERFORMANCE_COMPARISON.items():
        print(f"\n【{config_name}】")
        print(f"  训练时间: {details['training_time']}")
        print(f"  预期mAP: {details['expected_map']}")
        print(f"  模型大小: {details['model_size']}")
        print(f"  成本: {details['cost']}")
        print(f"  优点: {', '.join(details['pros'])}")
        print(f"  缺点: {', '.join(details['cons'])}")
    
    print("\n\n" + "=" * 80)
    print("✅ 使用建议:")
    print("  1. 本地M1: 快速测试、验证流程 (train_yolo.py)")
    print("  2. 云端T4: 正式训练、获得高质量模型 (train_yolo_cloud.py)")
    print("  3. 下载模型: 在本地部署和使用")
    print("=" * 80)
