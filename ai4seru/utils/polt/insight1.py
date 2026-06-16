import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib参数以支持学术论文风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16  # 从12改为16
plt.rcParams['axes.labelsize'] = 18  # 从14改为18
plt.rcParams['axes.titlesize'] = 18  # 从14改为18
plt.rcParams['legend.fontsize'] = 15  # 从11改为15
plt.rcParams['xtick.labelsize'] = 16  # 从12改为16
plt.rcParams['ytick.labelsize'] = 16  # 从12改为16
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


def plot_normalized_ranks():
    """
    绘制归一化排名对比图（第一张图）
    展示Fitness Rank和Slack Time Rank的归一化值
    """
    # 读取Excel文件
    df = pd.read_excel('csv/scheduling_analysis.xlsx')

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制两条线
    ax.plot(df['step'], df['normalized_fitness_rank'],
            color='tab:blue', linewidth=2, label=r'Normalized seru-batch compatibility Rank')
    ax.plot(df['step'], df['normalized_slack_rank'],
            color='tab:orange', linewidth=2, label='Normalized slack time rank')

    # 设置坐标轴
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Rank', fontsize=14, fontweight='bold')
    ax.set_xlim([df['step'].min(), df['step'].max()])
    ax.set_ylim([0, 1.25])  # 归一化值在0-1之间

    # 设置图例
    ax.legend(loc='upper right', frameon=True)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局
    plt.tight_layout()

    # 保存图形
    os.makedirs('fig', exist_ok=True)
    output_path = os.path.join('fig', 'normalized_ranks.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = os.path.join('fig', 'normalized_ranks.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Normalized ranks figure saved")

    plt.show()


def plot_slack_time():
    """
    绘制Selected Batch Slack Time（第二张图）
    """
    df = pd.read_excel('csv/scheduling_analysis.xlsx')

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制slack time
    ax.plot(df['step'], df['selected_batch_slack_time'],
            color='tab:green', linewidth=2, label='Selected Batch Slack Time')
    ax.fill_between(df['step'], df['selected_batch_slack_time'],
                    alpha=0.3, color='tab:green')

    # 设置坐标轴
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Slack Time', fontsize=14, fontweight='bold')
    ax.set_xlim([df['step'].min(), df['step'].max()])

    # 设置图例（可选，因为只有一条线）
    ax.legend(loc='upper right', frameon=True)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局
    plt.tight_layout()

    # 保存图形
    output_path = os.path.join('fig', 'selected_batch_slack_time.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = os.path.join('fig', 'selected_batch_slack_time.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Slack time figure saved")

    plt.show()


def plot_fitness_score():
    """
    绘制Selected Batch Fitness（第三张图）
    """
    df = pd.read_excel('csv/scheduling_analysis.xlsx')

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制fitness score
    ax.plot(df['step'], df['selected_batch_fitness'],
            color='tab:purple', linewidth=2, label='Selected Batch Fitness')
    ax.fill_between(df['step'], df['selected_batch_fitness'],
                    alpha=0.3, color='tab:purple')

    # 设置坐标轴
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness Score', fontsize=14, fontweight='bold')
    ax.set_xlim([df['step'].min(), df['step'].max()])
    ax.set_ylim([0, 1.05])  # Fitness在0-1之间

    # 设置图例（可选）
    ax.legend(loc='lower right', frameon=True)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局
    plt.tight_layout()

    # 保存图形
    output_path = os.path.join('fig', 'selected_batch_fitness.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = os.path.join('fig', 'selected_batch_fitness.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Fitness figure saved")

    plt.show()


def analyze_tradeoff_statistics():
    """
    分析权衡统计数据
    """
    df = pd.read_excel('csv/scheduling_analysis.xlsx')

    print("\n=== Trade-off Analysis Statistics ===\n")

    # 分析fitness rank的稳定性
    print(f"Fitness Rank Statistics:")
    print(f"  Mean normalized rank: {df['normalized_fitness_rank'].mean():.4f}")
    print(f"  Std normalized rank: {df['normalized_fitness_rank'].std():.4f}")
    print(f"  % of times selecting top-3 fitness: {(df['fitness_rank'] <= 3).mean() * 100:.1f}%")

    print(f"\nSlack Time Rank Statistics:")
    print(f"  Mean normalized rank: {df['normalized_slack_rank'].mean():.4f}")
    print(f"  Std normalized rank: {df['normalized_slack_rank'].std():.4f}")
    print(f"  % of times selecting most urgent: {(df['slack_time_rank'] == 1).mean() * 100:.1f}%")

    print(f"\nTrade-off Behavior:")
    print(f"  Steps where fitness > urgency priority: {(df['normalized_fitness_rank'] < df['normalized_slack_rank']).sum()}")
    print(f"  Steps where urgency > fitness priority: {(df['normalized_slack_rank'] < df['normalized_fitness_rank']).sum()}")

    # 计算相关性
    correlation = df['normalized_fitness_rank'].corr(df['normalized_slack_rank'])
    print(f"  Correlation between ranks: {correlation:.4f}")


def check_data_format():
    """
    检查数据格式并显示数据概览
    """
    try:
        # 尝试读取Excel文件
        df = pd.read_excel('csv/scheduling_analysis.xlsx')

        print("=== Data Overview ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        # 检查必要的列是否存在
        required_columns = [
            'step', 'normalized_fitness_rank', 'normalized_slack_rank',
            'selected_batch_slack_time', 'selected_batch_fitness'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"\nWarning: Missing columns: {missing_columns}")
        else:
            print("\nAll required columns found!")

        return True

    except FileNotFoundError:
        print("Error: File 'csv/scheduling_analysis.xlsx' not found!")
        print("Please ensure the file exists in the csv/ folder")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


if __name__ == "__main__":
    # 首先检查数据格式
    print("Checking data file...")
    if not check_data_format():
        print("\nPlease fix the data file issue before running the plots.")
    else:
        # 绘制三张图
        print("\nPlotting normalized ranks comparison...")
        plot_normalized_ranks()

        print("\nPlotting selected batch slack time...")
        plot_slack_time()

        print("\nPlotting selected batch fitness...")
        plot_fitness_score()

        # 分析统计数据
        analyze_tradeoff_statistics()