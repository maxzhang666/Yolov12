"""
YOLO12 模型导出脚本
支持导出 ONNX 和 TensorRT 格式，支持 INT8 量化

使用方式：
  # 导出 ONNX (FP32)
  python export_model.py --model runs/detect/yolo12n_person_head/weights/best.pt --format onnx
  
  # 导出 ONNX (INT8)
  python export_model.py --model runs/detect/yolo12n_person_head/weights/best.pt --format onnx --int8
  
  # 导出 ONNX (INT8) 指定数据集目录
  python export_model.py --model runs/detect/yolo12n_person_head/weights/best.pt --format onnx --int8 --dataset-dir DF-Data
  
  # 导出 TensorRT (FP16)
  python export_model.py --model runs/detect/yolo12n_person_head/weights/best.pt --format engine
  
  # 导出 TensorRT (INT8)
  python export_model.py --model runs/detect/yolo12n_person_head/weights/best.pt --format engine --int8
  
  # 导出多种格式
  python export_model.py --model runs/detect/yolo12n_person_head/weights/best.pt --format onnx,engine --int8
"""

from ultralytics import YOLO
import argparse
import os


def export_model(model_path, formats, int8=False, half=False, imgsz=640, data_yaml=None, dataset_dir='datasets'):
    """
    导出模型
    
    Args:
        model_path: 模型路径
        formats: 导出格式列表 ['onnx', 'engine', 'torchscript', 等]
        int8: 是否使用 INT8 量化
        half: 是否使用 FP16（仅 TensorRT）
        imgsz: 图像尺寸
        data_yaml: 数据集配置（INT8 量化需要）
        dataset_dir: 数据集目录（默认 'datasets'）
    """
    print("=" * 70)
    print("🚀 YOLO12 模型导出")
    print("=" * 70)
    print(f"📁 模型: {model_path}")
    print(f"📦 格式: {', '.join(formats)}")
    print(f"📐 图像尺寸: {imgsz}")
    if int8:
        print(f"⚡ INT8 量化: 启用")
        print(f"📊 校准数据: {data_yaml}")
    if half:
        print(f"⚡ FP16: 启用")
    print("=" * 70)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # INT8 量化需要数据集
    if int8 and not data_yaml:
        data_yaml = os.path.join(dataset_dir, 'data.yaml')
        print(f"\n⚠️  警告: INT8 量化需要数据集用于校准")
        print(f"   使用默认数据集: {data_yaml}")
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # 加载模型
    print("\n📥 加载模型...")
    model = YOLO(model_path)
    print("✅ 模型加载成功")
    
    # 导出每种格式
    for fmt in formats:
        print(f"\n🔄 导出 {fmt.upper()} 格式...")
        
        try:
            export_args = {
                'format': fmt,
                'imgsz': imgsz,
            }
            
            # ONNX 特定参数
            if fmt == 'onnx':
                export_args['simplify'] = True
                export_args['opset'] = 12
                if int8:
                    export_args['int8'] = True
                    export_args['data'] = data_yaml
            
            # TensorRT 特定参数
            elif fmt == 'engine':
                export_args['half'] = half if not int8 else False
                if int8:
                    export_args['int8'] = True
                    export_args['data'] = data_yaml
            
            # 执行导出
            export_path = model.export(**export_args)
            
            print(f"✅ {fmt.upper()} 导出成功: {export_path}")
            
            # 显示文件大小
            if os.path.exists(export_path):
                file_size = os.path.getsize(export_path) / (1024 * 1024)
                print(f"   文件大小: {file_size:.2f} MB")
        
        except Exception as e:
            print(f"❌ {fmt.upper()} 导出失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 导出完成!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='YOLO12 模型导出')
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (best.pt)')
    
    # 导出格式
    parser.add_argument('--format', type=str, default='onnx',
                        help='导出格式，多个用逗号分隔 (onnx,engine,torchscript,coreml)')
    
    # 量化选项
    parser.add_argument('--int8', action='store_true',
                        help='启用 INT8 量化')
    parser.add_argument('--half', action='store_true',
                        help='启用 FP16 (仅 TensorRT)')
    
    # 其他参数
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集配置文件 (INT8 量化需要)')
    parser.add_argument('--dataset-dir', type=str, default='datasets',
                        help='数据集目录 (默认: datasets)')
    
    args = parser.parse_args()
    
    # 解析格式
    formats = [f.strip() for f in args.format.split(',')]
    
    # 导出模型
    export_model(
        model_path=args.model,
        formats=formats,
        int8=args.int8,
        half=args.half,
        imgsz=args.imgsz,
        data_yaml=args.data,
        dataset_dir=args.dataset_dir
    )


if __name__ == '__main__':
    print("\n📋 支持的导出格式:")
    print("  onnx       - ONNX (推荐，跨平台)")
    print("  engine     - TensorRT (NVIDIA GPU)")
    print("  torchscript- TorchScript (PyTorch)")
    print("  coreml     - CoreML (iOS/macOS)")
    print("  openvino   - OpenVINO (Intel)")
    print("\n📋 量化选项:")
    print("  --int8     - INT8 量化 (速度快，精度略降)")
    print("  --half     - FP16 半精度 (仅 TensorRT)")
    print()
    
    main()
