import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# 设置matplotlib参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def plot_skill_similarity_vs_tardiness():
    """
    绘制技能相似度与总延迟的散点图及回归线
    """
    # 读取worker.xlsx文件
    df = pd.read_excel('csv/worker.xlsx', header=None, names=['skill_similarity', 'total_tardiness'])

    # 如果数据是以文本形式存储的，需要分割
    if df.shape[1] == 1:
        data = df.iloc[:, 0].str.split(expand=True)
        df = pd.DataFrame({
            'skill_similarity': data[0].astype(float),
            'total_tardiness': data[1].astype(float)
        })

    fig, ax = plt.subplots(figsize=(10, 7))

    # 提取数据
    x = df['skill_similarity'].values
    y = df['total_tardiness'].values

    # 计算线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    # 生成回归线的点
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # 绘制散点图
    scatter = ax.scatter(x, y, alpha=0.7, s=80, color='tab:blue',
                         edgecolors='darkblue', linewidth=1)

    # 绘制回归线
    ax.plot(x_line, y_line, color='tab:red', linewidth=2.5,
            label=f'Linear Regression (R² = {r_squared:.4f})', zorder=5)

    # 添加95%置信区间
    y_pred = slope * x + intercept
    residuals = y - y_pred
    std_error = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2))
    confidence = 1.96 * std_error
    ax.fill_between(x_line, y_line - confidence, y_line + confidence,
                    alpha=0.15, color='tab:red')

    # 设置标签
    ax.set_xlabel('Average Skill Similarity within Seru Formation',
                  fontsize=18, fontweight='bold')
    ax.set_ylabel('Total Tardiness', fontsize=18, fontweight='bold')

    # 设置坐标轴范围
    x_margin = (x.max() - x.min()) * 0.05
    y_margin = (y.max() - y.min()) * 0.05
    ax.set_xlim([x.min() - x_margin, x.max() + x_margin])
    ax.set_ylim([min(0, y.min() - y_margin), y.max() + y_margin])

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 添加图例 - 放在左上角避免与数据点重合
    ax.legend(loc='upper left', frameon=True, shadow=False, fontsize=15)

    # 添加回归方程和统计信息 - 放在右下角
    if slope < 0:
        equation_text = f'y = {slope:.1f}x + {intercept:.1f}'
    else:
        equation_text = f'y = {slope:.1f}x + {intercept:.1f}'

    stats_text = f'{equation_text}\np-value = {p_value:.4f}'

    # 文本框放在右下角，避免与图例重合
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    # 设置y轴格式为科学计数法（如果数值很大）
    if y.max() > 10000:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # 调整布局
    plt.tight_layout()

    # 保存图形
    os.makedirs('fig', exist_ok=True)
    output_path = os.path.join('fig', 'skill_similarity_vs_tardiness.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_png = os.path.join('fig', 'skill_similarity_vs_tardiness.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')

    print(f"Skill similarity vs tardiness figure saved to {output_path}")

    # 显示图形
    plt.show()

    # 打印详细统计信息
    print(f"\n=== Regression Analysis Results ===")
    print(f"Number of data points: {len(x)}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Correlation coefficient (r): {r_value:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Slope: {slope:.2f}")
    print(f"Intercept: {intercept:.2f}")

    return r_squared, slope, intercept


# Alternative layout with separate legend and equation placement.
def plot_skill_similarity_vs_tardiness_alt():
    """
    备选版本：只显示图例，不显示方程文本框
    """
    df = pd.read_excel('csv/worker.xlsx', header=None, names=['skill_similarity', 'total_tardiness'])

    if df.shape[1] == 1:
        data = df.iloc[:, 0].str.split(expand=True)
        df = pd.DataFrame({
            'skill_similarity': data[0].astype(float),
            'total_tardiness': data[1].astype(float)
        })

    fig, ax = plt.subplots(figsize=(10, 7))

    x = df['skill_similarity'].values
    y = df['total_tardiness'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # 散点图
    ax.scatter(x, y, alpha=0.7, s=80, color='tab:blue',
               edgecolors='darkblue', linewidth=1, label='Data points')

    # 回归线 - 在标签中包含所有信息
    regression_label = f'Linear Regression\nR² = {r_squared:.4f}\ny = {slope:.1f}x + {intercept:.1f}'
    ax.plot(x_line, y_line, color='tab:red', linewidth=2.5,
            label=regression_label, zorder=5)

    # 置信区间
    y_pred = slope * x + intercept
    residuals = y - y_pred
    std_error = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2))
    confidence = 1.96 * std_error
    ax.fill_between(x_line, y_line - confidence, y_line + confidence,
                    alpha=0.15, color='tab:red', label='95% CI')

    ax.set_xlabel('Average skill of workers similarity within seru formation',
                  fontsize=18, fontweight='bold')
    ax.set_ylabel('Total tardiness', fontsize=18, fontweight='bold')

    x_margin = (x.max() - x.min()) * 0.05
    y_margin = (y.max() - y.min()) * 0.05
    ax.set_xlim([x.min() - x_margin, x.max() + x_margin])
    ax.set_ylim([min(0, y.min() - y_margin), y.max() + y_margin])

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 图例放在最佳位置，matplotlib会自动找不遮挡数据的位置
    ax.legend(loc='best', frameon=True, shadow=False, fontsize=15)

    if y.max() > 10000:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()

    os.makedirs('fig', exist_ok=True)
    output_path = os.path.join('fig', 'skill_similarity_vs_tardiness_alt.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Alternative figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    try:
        # Layout with the legend at upper left and the equation at lower right.
        print("Creating main version...")
        plot_skill_similarity_vs_tardiness()

        # Layout with all information in the legend.
        print("\nCreating alternative version...")
        plot_skill_similarity_vs_tardiness_alt()

    except FileNotFoundError:
        print("Error: File 'csv/worker.xlsx' not found!")
    except Exception as e:
        print(f"Error: {e}")
