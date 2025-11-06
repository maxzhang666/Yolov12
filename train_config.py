"""
训练配置文件
可以通过修改这个文件来调整训练参数，无需修改主训练脚本
"""

class TrainConfig:
    """训练配置类"""
    
    # ============ 基础配置 ============
    DATA_YAML = 'datasets/data.yaml'          # 数据集配置文件
    MODEL_NAME = 'yolo12n.pt'                # 模型选择
    
    # ============ 训练参数 ============
    EPOCHS = 50                              # 训练轮数
    BATCH_SIZE = 16                          # 批次大小 (M1建议: 8-16)
    IMG_SIZE = 640                           # 图像尺寸
    WORKERS = 4                              # 数据加载线程数
    PATIENCE = 20                            # 早停耐心值
    
    # ============ 设备配置 ============
    # 'mps' - Apple Silicon加速
    # 'cpu' - CPU训练
    # 'cuda' - NVIDIA GPU (未来使用)
    DEVICE = 'mps'                           # M1默认使用MPS
    
    # ============ 保存配置 ============
    PROJECT_NAME = 'runs/detect'             # 项目文件夹
    EXPERIMENT_NAME = 'yolo12n_person_head'  # 实验名称
    SAVE_PERIOD = 10                         # 每N个epoch保存一次
    
    # ============ 优化配置 ============
    OPTIMIZER = 'auto'                       # 优化器 (auto/SGD/Adam/AdamW)
    LR0 = 0.01                              # 初始学习率
    LRF = 0.01                              # 最终学习率 (lr0 * lrf)
    MOMENTUM = 0.937                         # SGD动量/Adam beta1
    WEIGHT_DECAY = 0.0005                    # 权重衰减
    
    # ============ 数据增强 ============
    HSV_H = 0.015                           # 色调增强
    HSV_S = 0.7                             # 饱和度增强
    HSV_V = 0.4                             # 亮度增强
    DEGREES = 0.0                           # 旋转角度 (±deg)
    TRANSLATE = 0.1                         # 平移 (±fraction)
    SCALE = 0.5                             # 缩放 (gain)
    SHEAR = 0.0                             # 剪切 (±deg)
    PERSPECTIVE = 0.0                        # 透视变换 (±fraction)
    FLIPUD = 0.0                            # 上下翻转概率
    FLIPLR = 0.5                            # 左右翻转概率
    MOSAIC = 1.0                            # Mosaic增强概率
    MIXUP = 0.0                             # Mixup增强概率
    
    # ============ 高级配置 ============
    AMP = True                              # 自动混合精度训练
    PRETRAINED = True                        # 使用预训练权重
    VERBOSE = True                          # 详细输出
    SEED = 42                               # 随机种子
    DETERMINISTIC = True                     # 确定性训练
    PLOTS = True                            # 生成训练图表
    VAL = True                              # 训练时进行验证


# ============ 不同场景的预设配置 ============

class QuickTestConfig(TrainConfig):
    """快速测试配置 - 用于验证流程"""
    EPOCHS = 5
    BATCH_SIZE = 8
    SAVE_PERIOD = 2
    EXPERIMENT_NAME = 'quick_test'


class StandardConfig(TrainConfig):
    """标准配置 - 推荐使用"""
    EPOCHS = 50
    BATCH_SIZE = 16
    EXPERIMENT_NAME = 'standard_train'


class HighQualityConfig(TrainConfig):
    """高质量配置 - 追求更好效果"""
    EPOCHS = 100
    BATCH_SIZE = 16
    PATIENCE = 30
    EXPERIMENT_NAME = 'high_quality'
    
    # 更强的数据增强
    MOSAIC = 1.0
    MIXUP = 0.1


class M1OptimizedConfig(TrainConfig):
    """M1优化配置 - 适合MacBook Air M1"""
    EPOCHS = 50
    BATCH_SIZE = 8              # 降低批次大小防止内存溢出
    WORKERS = 4
    AMP = True                  # 启用混合精度加速
    EXPERIMENT_NAME = 'm1_optimized'


# ============ 使用说明 ============
"""
使用方法:

1. 快速测试:
   config = QuickTestConfig()

2. 标准训练:
   config = StandardConfig()

3. 高质量训练:
   config = HighQualityConfig()

4. M1优化训练:
   config = M1OptimizedConfig()

或者自定义:
   config = TrainConfig()
   config.EPOCHS = 80
   config.BATCH_SIZE = 12
"""
