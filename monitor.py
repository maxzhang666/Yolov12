"""
训练监控工具
用于实时查看训练进度和性能指标
"""

import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def watch_training(project_dir='runs/detect/yolo12n_person_head', interval=5):
    """
    实时监控训练进度
    
    Args:
        project_dir: 训练结果目录
        interval: 刷新间隔（秒）
    """
    results_csv = os.path.join(project_dir, 'results.csv')
    
    print("=" * 60)
    print("📊 YOLO12 训练监控")
    print("=" * 60)
    print(f"监控目录: {project_dir}")
    print(f"刷新间隔: {interval}秒")
    print("按 Ctrl+C 停止监控")
    print("=" * 60)
    
    last_epoch = -1
    
    try:
        while True:
            if not os.path.exists(results_csv):
                print("⏳ 等待训练开始...")
                time.sleep(interval)
                continue
            
            # 读取结果
            try:
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()  # 去除列名空格
                
                current_epoch = len(df)
                
                if current_epoch > last_epoch:
                    last_epoch = current_epoch
                    
                    # 获取最新数据
                    latest = df.iloc[-1]
                    
                    # 清屏（可选）
                    # os.system('clear')
                    
                    print("\n" + "=" * 60)
                    print(f"📈 Epoch {current_epoch}/{latest.get('epoch', 'N/A')}")
                    print("=" * 60)
                    
                    # 训练指标
                    print("\n🏋️  训练指标:")
                    print(f"  Box Loss:  {latest.get('train/box_loss', 0):.4f}")
                    print(f"  Cls Loss:  {latest.get('train/cls_loss', 0):.4f}")
                    print(f"  DFL Loss:  {latest.get('train/dfl_loss', 0):.4f}")
                    
                    # 验证指标
                    print("\n✅ 验证指标:")
                    print(f"  Precision: {latest.get('metrics/precision(B)', 0):.4f}")
                    print(f"  Recall:    {latest.get('metrics/recall(B)', 0):.4f}")
                    print(f"  mAP50:     {latest.get('metrics/mAP50(B)', 0):.4f}")
                    print(f"  mAP50-95:  {latest.get('metrics/mAP50-95(B)', 0):.4f}")
                    
                    # 学习率
                    if 'lr/pg0' in latest:
                        print(f"\n📊 学习率: {latest['lr/pg0']:.6f}")
                    
                    # 最佳结果
                    best_map = df['metrics/mAP50-95(B)'].max()
                    best_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
                    print(f"\n🏆 最佳 mAP50-95: {best_map:.4f} (Epoch {best_epoch})")
                    
                    print("=" * 60)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  读取结果时出错: {e}")
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  监控已停止")


def plot_training_curves(project_dir='runs/detect/yolo12n_person_head'):
    """
    绘制训练曲线
    
    Args:
        project_dir: 训练结果目录
    """
    results_csv = os.path.join(project_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print(f"❌ 找不到结果文件: {results_csv}")
        return
    
    # 读取数据
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO12 训练曲线', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
    ax1.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', linewidth=2)
    ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. mAP曲线
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color='green')
    ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2, color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title('平均精度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 精确率和召回率
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='orange')
    ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='purple')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('精确率与召回率')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 学习率
    ax4 = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax4.plot(df['epoch'], df['lr/pg0'], linewidth=2, color='red')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('学习率变化')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(project_dir, 'training_curves_custom.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {save_path}")
    
    plt.show()


def summary_results(project_dir='runs/detect/yolo12n_person_head'):
    """
    输出训练结果摘要
    
    Args:
        project_dir: 训练结果目录
    """
    results_csv = os.path.join(project_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print(f"❌ 找不到结果文件: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    print("\n" + "=" * 60)
    print("📊 训练结果摘要")
    print("=" * 60)
    
    print(f"\n📝 基本信息:")
    print(f"  总训练轮数: {len(df)}")
    print(f"  项目目录: {project_dir}")
    
    print(f"\n🏆 最佳结果:")
    best_map50_95 = df['metrics/mAP50-95(B)'].max()
    best_epoch_95 = df['metrics/mAP50-95(B)'].idxmax() + 1
    best_map50 = df['metrics/mAP50(B)'].max()
    
    print(f"  最佳 mAP50-95: {best_map50_95:.4f} (Epoch {best_epoch_95})")
    print(f"  最佳 mAP50:    {best_map50:.4f}")
    print(f"  最佳 Precision: {df['metrics/precision(B)'].max():.4f}")
    print(f"  最佳 Recall:    {df['metrics/recall(B)'].max():.4f}")
    
    print(f"\n📉 最终结果 (Epoch {len(df)}):")
    final = df.iloc[-1]
    print(f"  mAP50-95:  {final['metrics/mAP50-95(B)']:.4f}")
    print(f"  mAP50:     {final['metrics/mAP50(B)']:.4f}")
    print(f"  Precision: {final['metrics/precision(B)']:.4f}")
    print(f"  Recall:    {final['metrics/recall(B)']:.4f}")
    
    print(f"\n📁 模型文件:")
    weights_dir = os.path.join(project_dir, 'weights')
    if os.path.exists(weights_dir):
        best_pt = os.path.join(weights_dir, 'best.pt')
        last_pt = os.path.join(weights_dir, 'last.pt')
        
        if os.path.exists(best_pt):
            size_mb = os.path.getsize(best_pt) / (1024 * 1024)
            print(f"  ✅ best.pt ({size_mb:.1f} MB)")
        
        if os.path.exists(last_pt):
            size_mb = os.path.getsize(last_pt) / (1024 * 1024)
            print(f"  ✅ last.pt ({size_mb:.1f} MB)")
    
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    # 默认项目目录
    project_dir = 'runs/detect/yolo12n_person_head'
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'summary'
    
    if mode == 'watch':
        # 实时监控
        watch_training(project_dir)
    elif mode == 'plot':
        # 绘制曲线
        plot_training_curves(project_dir)
    elif mode == 'summary':
        # 输出摘要
        summary_results(project_dir)
    else:
        print("使用方法:")
        print("  python monitor.py [mode]")
        print("")
        print("模式:")
        print("  summary  - 显示训练结果摘要（默认）")
        print("  watch    - 实时监控训练进度")
        print("  plot     - 绘制训练曲线")
