import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

# 设置matplotlib参数以支持学术论文风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


def smooth_curve(values, window_length=51, polyorder=3):
    """
    使用Savitzky-Golay滤波器平滑曲线
    window_length: 窗口长度，必须是奇数
    polyorder: 多项式阶数
    """
    if len(values) < window_length:
        window_length = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window_length <= polyorder:
        return values
    return savgol_filter(values, window_length, polyorder)


def plot_fps_curves():
    """
    绘制不同环境数量下的FPS曲线
    """
    # 定义文件名和对应的环境数量
    env_configs = [
        ('1env_fps.csv', '1 Environment', 'tab:blue'),
        ('2env_fps.csv', '2 Environments', 'tab:orange'),
        ('4env_fps.csv', '4 Environments', 'tab:green'),
        ('8env_fps.csv', '8 Environments', 'tab:red'),
        ('16env_fps.csv', '16 Environments', 'tab:purple')
    ]

    # 创建输出目录（如果不存在）
    os.makedirs('fig', exist_ok=True)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 读取并绘制每个文件的数据
    for filename, label, color in env_configs:
        filepath = os.path.join('csv', filename)

        try:
            # 读取CSV文件
            df = pd.read_csv(filepath)

            # 确保列名正确（处理可能的空格问题）
            df.columns = df.columns.str.strip()

            # 提取Step和Value列（FPS值）
            steps = df['Step'].values
            fps_values = df['Value'].values

            # 对数据进行降采样以提高可视化效果（如果数据点太多）
            if len(steps) > 500:
                # 每隔n个点取一个
                sample_rate = len(steps) // 500
                steps_sampled = steps[::sample_rate]
                fps_sampled = fps_values[::sample_rate]
            else:
                steps_sampled = steps
                fps_sampled = fps_values

            # 平滑曲线（FPS数据可能波动较大，需要更强的平滑）
            fps_smooth = smooth_curve(fps_sampled, window_length=31, polyorder=3)

            # 绘制原始数据（淡色）
            ax.plot(steps_sampled, fps_sampled, color=color, alpha=0.2, linewidth=0.5)

            # 绘制平滑后的曲线（实线）
            ax.plot(steps_sampled, fps_smooth, color=color, label=label, linewidth=2)

        except FileNotFoundError:
            print(f"Warning: File {filepath} not found. Skipping...")
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")

    # 设置坐标轴标签
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frames Per Second (FPS)', fontsize=14, fontweight='bold')

    # 设置图例
    ax.legend(loc='best', frameon=True, shadow=True, ncol=1)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置y轴从0开始（FPS通常是正值）
    ax.set_ylim(bottom=0)

    # 添加更好的刻度
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # 调整布局
    plt.tight_layout()

    # 保存图形
    output_path = os.path.join('fig', 'fps_curves.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # 也保存为PNG格式以便预览
    output_path_png = os.path.join('fig', 'fps_curves.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Figure also saved to {output_path_png}")

    # 显示图形
    plt.show()


def analyze_fps_stats():
    """
    分析并打印FPS统计信息
    """
    env_configs = [
        ('1env_fps.csv', '1 Environment'),
        ('2env_fps.csv', '2 Environments'),
        ('4env_fps.csv', '4 Environments'),
        ('8env_fps.csv', '8 Environments'),
        ('16env_fps.csv', '16 Environments')
    ]

    print("\n=== FPS Analysis ===\n")

    fps_summary = []

    for filename, label in env_configs:
        filepath = os.path.join('csv', filename)

        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()

            fps_values = df['Value'].values

            # 计算统计信息
            mean_fps = np.mean(fps_values)
            max_fps = np.max(fps_values)
            min_fps = np.min(fps_values)
            std_fps = np.std(fps_values)
            # 稳定后的FPS（取后50%的数据）
            stable_fps = np.mean(fps_values[len(fps_values) // 2:])

            print(f"{label}:")
            print(f"  Mean FPS: {mean_fps:.2f}")
            print(f"  Max FPS: {max_fps:.2f}")
            print(f"  Min FPS: {min_fps:.2f}")
            print(f"  Std FPS: {std_fps:.2f}")
            print(f"  Stable FPS (last 50%): {stable_fps:.2f}")
            print()

            # 保存用于对比
            env_num = int(filename.split('env')[0])
            fps_summary.append((env_num, stable_fps))

        except Exception as e:
            print(f"Error analyzing {filepath}: {str(e)}")

    # 计算并打印加速比
    if fps_summary:
        print("=== Speedup Analysis ===\n")
        fps_summary.sort(key=lambda x: x[0])
        base_fps = fps_summary[0][1]  # 1 environment as baseline

        for env_num, fps in fps_summary:
            speedup = fps / base_fps
            print(f"{env_num:2d} Environment(s): {fps:8.2f} FPS, Speedup: {speedup:.2f}x")


def plot_fps_bar_chart():
    """
    绘制FPS柱状图对比（可选）
    """
    env_configs = [
        ('1env_fps.csv', '1'),
        ('2env_fps.csv', '2'),
        ('4env_fps.csv', '4'),
        ('8env_fps.csv', '8'),
        ('16env_fps.csv', '16')
    ]

    env_nums = []
    mean_fps = []

    for filename, env_label in env_configs:
        filepath = os.path.join('csv', filename)
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            fps_values = df['Value'].values

            # 使用稳定后的平均FPS（后50%的数据）
            stable_fps = np.mean(fps_values[len(fps_values) // 2:])
            env_nums.append(env_label)
            mean_fps.append(stable_fps)
        except:
            continue

    if env_nums and mean_fps:
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        bars = ax.bar(env_nums, mean_fps, color=colors[:len(env_nums)], alpha=0.8)

        # 在柱子上添加数值
        for bar, fps in zip(bars, mean_fps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{fps:.0f}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Number of Environments', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average FPS', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        plt.tight_layout()

        output_path = os.path.join('fig', 'fps_bar_chart.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to {output_path}")

        plt.show()


if __name__ == "__main__":
    # 绘制FPS曲线
    print("Plotting FPS curves...")
    plot_fps_curves()

    # 分析FPS统计
    analyze_fps_stats()

    # 绘制FPS柱状图（可选）
    print("\nPlotting FPS bar chart...")
    plot_fps_bar_chart()