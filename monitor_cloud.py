"""
云服务器训练监控工具 (T4 GPU)
除了训练指标，还监控 GPU 使用情况
"""

import os
import time
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def get_gpu_info():
    """获取 GPU 使用信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(gpu_util),
                'mem_used': int(mem_used),
                'mem_total': int(mem_total),
                'temp': int(temp),
                'mem_percent': int(mem_used) / int(mem_total) * 100
            }
    except:
        pass
    
    return None


def watch_training_cloud(project_dir='runs/detect/yolo12s_person_head_t4', interval=5):
    """
    实时监控云服务器训练进度（包括GPU）
    
    Args:
        project_dir: 训练结果目录
        interval: 刷新间隔（秒）
    """
    results_csv = os.path.join(project_dir, 'results.csv')
    
    print("=" * 70)
    print("🚀 YOLO12 云服务器训练监控 (T4 GPU)")
    print("=" * 70)
    print(f"监控目录: {project_dir}")
    print(f"刷新间隔: {interval}秒")
    print("按 Ctrl+C 停止监控")
    print("=" * 70)
    
    last_epoch = -1
    
    try:
        while True:
            # 获取GPU信息
            gpu_info = get_gpu_info()
            
            if not os.path.exists(results_csv):
                print(f"\n⏳ 等待训练开始... [{time.strftime('%H:%M:%S')}]")
                if gpu_info:
                    print(f"   GPU使用: {gpu_info['gpu_util']}% | "
                          f"显存: {gpu_info['mem_used']}/{gpu_info['mem_total']}MB "
                          f"({gpu_info['mem_percent']:.1f}%) | "
                          f"温度: {gpu_info['temp']}°C")
                time.sleep(interval)
                continue
            
            # 读取训练结果
            try:
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                
                current_epoch = len(df)
                
                if current_epoch > last_epoch:
                    last_epoch = current_epoch
                    latest = df.iloc[-1]
                    
                    # 清屏（可选）
                    # os.system('clear')
                    
                    print("\n" + "=" * 70)
                    print(f"📈 Epoch {current_epoch} | {time.strftime('%H:%M:%S')}")
                    print("=" * 70)
                    
                    # GPU信息
                    if gpu_info:
                        print(f"\n💻 GPU状态:")
                        print(f"  利用率: {gpu_info['gpu_util']}% ", end="")
                        if gpu_info['gpu_util'] > 80:
                            print("✅")
                        elif gpu_info['gpu_util'] > 50:
                            print("⚠️  (可以提高)")
                        else:
                            print("❌ (利用率低)")
                        
                        print(f"  显存: {gpu_info['mem_used']}/{gpu_info['mem_total']}MB "
                              f"({gpu_info['mem_percent']:.1f}%)")
                        print(f"  温度: {gpu_info['temp']}°C ", end="")
                        if gpu_info['temp'] < 75:
                            print("✅")
                        elif gpu_info['temp'] < 85:
                            print("⚠️")
                        else:
                            print("🔥 (温度高)")
                    
                    # 训练指标
                    print(f"\n🏋️  训练损失:")
                    print(f"  Box: {latest.get('train/box_loss', 0):.4f} | "
                          f"Cls: {latest.get('train/cls_loss', 0):.4f} | "
                          f"DFL: {latest.get('train/dfl_loss', 0):.4f}")
                    
                    # 验证指标
                    print(f"\n✅ 验证指标:")
                    precision = latest.get('metrics/precision(B)', 0)
                    recall = latest.get('metrics/recall(B)', 0)
                    map50 = latest.get('metrics/mAP50(B)', 0)
                    map50_95 = latest.get('metrics/mAP50-95(B)', 0)
                    
                    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f}")
                    print(f"  mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} ", end="")
                    
                    # mAP评级
                    if map50_95 > 0.7:
                        print("🌟 优秀!")
                    elif map50_95 > 0.6:
                        print("✅ 很好!")
                    elif map50_95 > 0.5:
                        print("👍 不错")
                    else:
                        print("")
                    
                    # 学习率
                    if 'lr/pg0' in latest:
                        print(f"\n📊 学习率: {latest['lr/pg0']:.6f}")
                    
                    # 最佳结果
                    best_map = df['metrics/mAP50-95(B)'].max()
                    best_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
                    print(f"\n🏆 最佳 mAP50-95: {best_map:.4f} (Epoch {best_epoch})")
                    
                    # 预计剩余时间（简单估算）
                    if current_epoch > 1:
                        total_epochs = latest.get('epoch', current_epoch)
                        if isinstance(total_epochs, (int, float)):
                            remaining = int(total_epochs) - current_epoch
                            print(f"⏱️  剩余: ~{remaining} epochs")
                    
                    print("=" * 70)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  读取结果时出错: {e}")
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  监控已停止")


def monitor_gpu_only(interval=1):
    """仅监控GPU使用情况"""
    print("=" * 70)
    print("💻 GPU 实时监控")
    print("=" * 70)
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            gpu_info = get_gpu_info()
            
            if gpu_info:
                print(f"\r[{time.strftime('%H:%M:%S')}] "
                      f"GPU: {gpu_info['gpu_util']:3d}% | "
                      f"显存: {gpu_info['mem_used']:5d}/{gpu_info['mem_total']}MB "
                      f"({gpu_info['mem_percent']:5.1f}%) | "
                      f"温度: {gpu_info['temp']:2d}°C", end="", flush=True)
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] ❌ 无法获取GPU信息", 
                      end="", flush=True)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  监控已停止")


if __name__ == '__main__':
    import sys
    
    # 默认项目目录（T4版本）
    project_dir = 'runs/detect/yolo12s_person_head_t4'
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'watch'
    
    if mode == 'watch':
        # 实时监控训练+GPU
        watch_training_cloud(project_dir)
    elif mode == 'gpu':
        # 仅监控GPU
        monitor_gpu_only()
    else:
        print("使用方法:")
        print("  python monitor_cloud.py [mode]")
        print("")
        print("模式:")
        print("  watch    - 监控训练进度 + GPU状态（默认）")
        print("  gpu      - 仅监控GPU状态")
