#!/usr/bin/env python3
"""
YOLO to Label Studio Converter
将YOLO格式的标注数据转换为Label Studio导入格式

Usage:
    python yolo2label_studio.py --dataset train --output output.json
    python yolo2label_studio.py --dataset valid --output output.json
    python yolo2label_studio.py --dataset test --output output.json
    python yolo2label_studio.py --dataset-path /custom/path --output output.json
"""

import json
import os
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import base64
from PIL import Image


class YoloToLabelStudioConverter:
    """YOLO格式到Label Studio格式的转换器"""
    
    def __init__(self, dataset_root: str, class_names: List[str]):
        """
        初始化转换器
        
        Args:
            dataset_root: 数据集根目录路径
            class_names: 类别名称列表
        """
        self.dataset_root = Path(dataset_root)
        self.class_names = class_names
        
    def read_yolo_annotation(self, label_file: Path) -> List[Dict[str, Any]]:
        """
        读取YOLO格式的标注文件
        
        Args:
            label_file: YOLO标注文件路径
            
        Returns:
            标注列表,每个标注包含class_id, x_center, y_center, width, height
        """
        annotations = []
        
        if not label_file.exists():
            return annotations
            
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 5:
                    annotations.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
                    
        return annotations
    
    def yolo_to_label_studio_bbox(self, yolo_box: Dict[str, Any], 
                                   img_width: int, img_height: int) -> Dict[str, float]:
        """
        将YOLO格式的边界框转换为Label Studio格式(百分比)
        
        Args:
            yolo_box: YOLO格式的box(归一化坐标)
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            Label Studio格式的box(百分比坐标)
        """
        # YOLO: x_center, y_center, width, height (0-1归一化)
        # Label Studio: x, y, width, height (百分比 0-100)
        
        x_center = yolo_box['x_center']
        y_center = yolo_box['y_center']
        width = yolo_box['width']
        height = yolo_box['height']
        
        # 转换为左上角坐标
        x = (x_center - width / 2) * 100
        y = (y_center - height / 2) * 100
        w = width * 100
        h = height * 100
        
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h
        }
    
    def convert_image(self, image_path: Path, label_path: Path, 
                     task_id: int, dataset_relative_path: str = None) -> Dict[str, Any]:
        """
        转换单个图像及其标注
        
        Args:
            image_path: 图像文件路径
            label_path: 标注文件路径
            task_id: 任务ID
            dataset_relative_path: 相对于datasets的路径 (如 "test/images")
            
        Returns:
            Label Studio任务字典
        """
        # 读取图像尺寸
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Warning: Cannot read image {image_path}: {e}")
            return None
        
        # 读取YOLO标注
        yolo_annotations = self.read_yolo_annotation(label_path)
        
        # 转换为Label Studio格式
        annotations = []
        for idx, yolo_box in enumerate(yolo_annotations):
            class_id = yolo_box['class_id']
            if class_id >= len(self.class_names):
                print(f"Warning: Class ID {class_id} out of range in {label_path}")
                continue
                
            ls_box = self.yolo_to_label_studio_bbox(yolo_box, img_width, img_height)
            
            annotation = {
                "id": f"{task_id}_{idx}",
                "type": "rectanglelabels",
                "value": {
                    "x": ls_box['x'],
                    "y": ls_box['y'],
                    "width": ls_box['width'],
                    "height": ls_box['height'],
                    "rotation": 0,
                    "rectanglelabels": [self.class_names[class_id]]
                },
                "to_name": "image",
                "from_name": "label",
                "image_rotation": 0,
                "original_width": img_width,
                "original_height": img_height
            }
            annotations.append(annotation)
        
        # 创建Label Studio任务
        # 使用相对于datasets的路径，格式：/data/local-files/?d=相对路径
        if dataset_relative_path:
            # 构建相对路径：如 test/images/xxx.jpg
            image_relative_path = f"{dataset_relative_path}/{image_path.name}"
        else:
            # 仅使用文件名
            image_relative_path = image_path.name
            
        image_url = f"/data/local-files/?d={image_relative_path}"
        
        task = {
            "data": {
                "image": image_url
            },
            "annotations": [
                {
                    "model_version": "yolo_converted",
                    "result": annotations
                }
            ] if annotations else []
        }
        
        return task
    
    def convert_dataset(self, images_dir: Path, labels_dir: Path, 
                       dataset_relative_path: str = None) -> List[Dict[str, Any]]:
        """
        转换整个数据集
        
        Args:
            images_dir: 图像目录路径
            labels_dir: 标签目录路径
            dataset_relative_path: 相对于datasets的路径 (如 "test/images")
            
        Returns:
            Label Studio任务列表
        """
        tasks = []
        task_id = 1
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 遍历所有图像
        if not images_dir.exists():
            print(f"Error: Images directory not found: {images_dir}")
            return tasks
            
        image_files = [f for f in images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images in {images_dir}")
        
        for image_file in sorted(image_files):
            # 查找对应的标注文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            # 转换
            task = self.convert_image(image_file, label_file, task_id, dataset_relative_path)
            if task:
                tasks.append(task)
                task_id += 1
                
                if task_id % 100 == 0:
                    print(f"Processed {task_id - 1} images...")
        
        print(f"Successfully converted {len(tasks)} images")
        return tasks


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    加载YOLO数据集配置文件
    
    Args:
        config_path: data.yaml配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Convert YOLO format annotations to Label Studio format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 转换训练集
  python yolo2label_studio.py --dataset train --output train_ls.json
  
  # 转换验证集
  python yolo2label_studio.py --dataset valid --output valid_ls.json
  
  # 转换测试集
  python yolo2label_studio.py --dataset test --output test_ls.json
  
  # 指定自定义数据集路径
  python yolo2label_studio.py --dataset-path ./datasets/custom --output custom_ls.json
  
  # 指定配置文件
  python yolo2label_studio.py --dataset train --config ./datasets/data.yaml --output train_ls.json
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'valid', 'test'],
        help='Dataset split to convert (train/valid/test)'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Custom dataset path (alternative to --dataset)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='./datasets/data.yaml',
        help='Path to data.yaml config file (default: ./datasets/data.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path for Label Studio'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default='./datasets',
        help='Project root directory (default: ./datasets)'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.dataset and not args.dataset_path:
        parser.error("Either --dataset or --dataset-path must be specified")
    
    if args.dataset and args.dataset_path:
        parser.error("Cannot specify both --dataset and --dataset-path")
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # 获取类别名称
    class_names = config.get('names', [])
    if not class_names:
        print("Error: No class names found in config")
        return
    
    print(f"Classes: {class_names}")
    
    # 确定数据集路径
    project_root = Path(args.project_root)
    
    if args.dataset_path:
        # 使用自定义路径
        dataset_path = Path(args.dataset_path)
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        # 计算相对于project_root的路径
        try:
            relative_path = dataset_path.relative_to(project_root)
            dataset_relative_path = f"{relative_path}/images"
        except ValueError:
            # 如果不在project_root下，使用数据集名称
            dataset_relative_path = f"{dataset_path.name}/images"
    else:
        # 使用标准数据集分割
        dataset_path = project_root / args.dataset
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        dataset_relative_path = f"{args.dataset}/images"
    
    print(f"Dataset path: {dataset_path}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Relative path for Label Studio: {dataset_relative_path}")
    
    # 检查目录是否存在
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"Warning: Labels directory not found: {labels_dir}")
        print("Creating tasks without annotations...")
    
    # 创建转换器
    converter = YoloToLabelStudioConverter(
        dataset_root=str(project_root),
        class_names=class_names
    )
    
    # 转换数据集
    print("\nStarting conversion...")
    tasks = converter.convert_dataset(images_dir, labels_dir, dataset_relative_path)
    
    if not tasks:
        print("No tasks were created. Please check your dataset.")
        return
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully converted {len(tasks)} tasks")
    print(f"✓ Output saved to: {output_path}")
    print(f"\nYou can now import {output_path} into Label Studio")
    
    # 打印统计信息
    total_annotations = sum(
        len(task['annotations'][0]['result']) 
        for task in tasks 
        if task.get('annotations')
    )
    print(f"\nStatistics:")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Average annotations per image: {total_annotations / len(tasks):.2f}")


if __name__ == '__main__':
    main()
