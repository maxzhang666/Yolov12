"""
YOLO12 模型推理测试脚本
用于测试训练好的模型
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path


def predict_image(model_path, image_path, save_dir='runs/predict', conf_threshold=0.25):
    """
    对单张图片进行预测
    
    Args:
        model_path: 模型权重路径
        image_path: 图片路径
        save_dir: 结果保存目录
        conf_threshold: 置信度阈值
    """
    print(f"🤖 加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"🖼️  预测图片: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='test',
        exist_ok=True
    )
    
    # 打印检测结果
    for result in results:
        boxes = result.boxes
        print(f"\n📊 检测到 {len(boxes)} 个目标:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = result.names[cls_id]
            print(f"  - {cls_name}: {conf:.2%}")
    
    print(f"\n✅ 结果已保存到: {save_dir}/test/")


def predict_folder(model_path, folder_path, save_dir='runs/predict', conf_threshold=0.25):
    """
    对文件夹中的所有图片进行预测
    
    Args:
        model_path: 模型权重路径
        folder_path: 图片文件夹路径
        save_dir: 结果保存目录
        conf_threshold: 置信度阈值
    """
    print(f"🤖 加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"📁 预测文件夹: {folder_path}")
    results = model.predict(
        source=folder_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='batch_test',
        exist_ok=True
    )
    
    print(f"\n✅ 处理了 {len(results)} 张图片")
    print(f"✅ 结果已保存到: {save_dir}/batch_test/")


def predict_video(model_path, video_path, save_dir='runs/predict', conf_threshold=0.25):
    """
    对视频进行预测
    
    Args:
        model_path: 模型权重路径
        video_path: 视频路径
        save_dir: 结果保存目录
        conf_threshold: 置信度阈值
    """
    print(f"🤖 加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"🎥 预测视频: {video_path}")
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='video_test',
        exist_ok=True,
        stream=True  # 流式处理，节省内存
    )
    
    # 处理视频帧
    frame_count = 0
    for result in results:
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次
            print(f"已处理 {frame_count} 帧...")
    
    print(f"\n✅ 视频处理完成，共 {frame_count} 帧")
    print(f"✅ 结果已保存到: {save_dir}/video_test/")


def evaluate_model(model_path, data_yaml):
    """
    在测试集上评估模型性能
    
    Args:
        model_path: 模型权重路径
        data_yaml: 数据集配置文件
    """
    print(f"🤖 加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"📊 在测试集上评估模型...")
    metrics = model.val(
        data=data_yaml,
        split='test',  # 使用测试集
        save_json=True,
        plots=True
    )
    
    print("\n" + "=" * 60)
    print("📊 测试集性能指标:")
    print("=" * 60)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    # ============ 配置 ============
    
    # 模型路径 (训练完成后的最佳模型)
    MODEL_PATH = 'runs/detect/yolo12n_person_head/weights/best.pt'
    
    # 数据集配置
    DATA_YAML = 'datasets/data.yaml'
    
    # 置信度阈值
    CONF_THRESHOLD = 0.25
    
    # ============ 选择测试模式 ============
    
    print("=" * 60)
    print("🔍 YOLO12 模型推理测试")
    print("=" * 60)
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请先运行 train_yolo.py 训练模型")
        exit(1)
    
    # 模式1: 在测试集上评估模型性能
    print("\n📊 模式1: 评估模型性能")
    evaluate_model(MODEL_PATH, DATA_YAML)
    
    # 模式2: 对测试集图片进行预测（可视化）
    print("\n🖼️  模式2: 测试集图片预测")
    test_images_path = 'datasets/test/images'
    if os.path.exists(test_images_path):
        predict_folder(
            MODEL_PATH, 
            test_images_path, 
            save_dir='runs/predict',
            conf_threshold=CONF_THRESHOLD
        )
    else:
        print(f"⚠️  测试集图片目录不存在: {test_images_path}")
    
    # 模式3: 对单张图片进行预测（示例）
    # 取消下面的注释来测试单张图片
    """
    print("\n🖼️  模式3: 单张图片预测")
    single_image = 'path/to/your/image.jpg'
    if os.path.exists(single_image):
        predict_image(
            MODEL_PATH,
            single_image,
            save_dir='runs/predict',
            conf_threshold=CONF_THRESHOLD
        )
    """
    
    # 模式4: 对视频进行预测（示例）
    # 取消下面的注释来测试视频
    """
    print("\n🎥 模式4: 视频预测")
    video_path = 'path/to/your/video.mp4'
    if os.path.exists(video_path):
        predict_video(
            MODEL_PATH,
            video_path,
            save_dir='runs/predict',
            conf_threshold=CONF_THRESHOLD
        )
    """
    
    print("\n✅ 所有测试完成!")
