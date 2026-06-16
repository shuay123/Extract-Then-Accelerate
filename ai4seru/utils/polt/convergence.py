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


def plot_convergence_curves():
    """
    绘制不同环境数量下的DRL收敛曲线
    """
    # 定义文件名和对应的环境数量
    env_configs = [
        ('1env_reward.csv', '1 Environment', 'tab:blue'),
        ('2env_reward.csv', '2 Environments', 'tab:orange'),
        ('4env_reward.csv', '4 Environments', 'tab:green'),
        ('8env_reward.csv', '8 Environments', 'tab:red'),
        ('16env_reward.csv', '16 Environments', 'tab:purple')
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

            # 提取Step和Value列
            steps = df['Step'].values
            values = df['Value'].values

            # 对数据进行降采样以提高可视化效果（如果数据点太多）
            if len(steps) > 500:
                # 每隔n个点取一个
                sample_rate = len(steps) // 500
                steps_sampled = steps[::sample_rate]
                values_sampled = values[::sample_rate]
            else:
                steps_sampled = steps
                values_sampled = values

            # 平滑曲线
            values_smooth = smooth_curve(values_sampled, window_length=21, polyorder=3)

            # 绘制原始数据（淡色）
            ax.plot(steps_sampled, values_sampled, color=color, alpha=0.3, linewidth=0.5)

            # 绘制平滑后的曲线（实线）
            ax.plot(steps_sampled, values_smooth, color=color, label=label, linewidth=2)

        except FileNotFoundError:
            print(f"Warning: File {filepath} not found. Skipping...")
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")

    # 设置坐标轴标签
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')

    # 设置图例
    ax.legend(loc='lower right', frameon=True, shadow=True, ncol=1)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置坐标轴范围（可以根据数据调整）
    # ax.set_xlim([0, None])  # 自动调整x轴范围
    # ax.set_ylim([None, 0])  # 假设reward是负值，最大为0

    # 添加更好的刻度
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # 调整布局
    plt.tight_layout()

    # 保存图形
    output_path = os.path.join('fig', 'convergence_curves.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # 也保存为PNG格式以便预览
    output_path_png = os.path.join('fig', 'convergence_curves.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Figure also saved to {output_path_png}")

    # 显示图形
    plt.show()


def analyze_convergence_stats():
    """
    分析并打印收敛统计信息
    """
    env_configs = [
        ('1env_reward.csv', '1 Environment'),
        ('2env_reward.csv', '2 Environments'),
        ('4env_reward.csv', '4 Environments'),
        ('8env_reward.csv', '8 Environments'),
        ('16env_reward.csv', '16 Environments')
    ]

    print("\n=== Convergence Analysis ===\n")

    for filename, label in env_configs:
        filepath = os.path.join('csv', filename)

        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()

            values = df['Value'].values

            # 计算统计信息
            final_reward = values[-1] if len(values) > 0 else 0
            max_reward = np.max(values)
            mean_last_100 = np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)
            std_last_100 = np.std(values[-100:]) if len(values) >= 100 else np.std(values)

            print(f"{label}:")
            print(f"  Final Reward: {final_reward:.2f}")
            print(f"  Max Reward: {max_reward:.2f}")
            print(f"  Mean (last 100 steps): {mean_last_100:.2f}")
            print(f"  Std (last 100 steps): {std_last_100:.2f}")
            print()

        except Exception as e:
            print(f"Error analyzing {filepath}: {str(e)}")


if __name__ == "__main__":
    # 绘制收敛曲线
    plot_convergence_curves()

    # 分析收敛统计
    analyze_convergence_stats()