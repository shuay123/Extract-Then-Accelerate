import pandas as pd
import glob

# 读取所有实例的汇总文件
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
filepath = "C:/code/datasets/Seru_datasets/JCompany/paper2025/instence/W8_J12/result/"
all_dfs = []
for i in range(1, 31):
    path = f"{filepath}/{i}resultAlpha.xlsx"
    df = pd.read_excel(path, sheet_name="result")
    df['instance'] = i
    all_dfs.append(df)

df_all = pd.concat(all_dfs)

# 重命名列（因为有重复列名，建议读取后手动指定）
df_all.columns = ['Alpha','flag_e','MIP_Cmax','MIP_NSF','MIP_Time',
                  'GNN_Cmax','gap_cmax','GNN_NSF','gap_nsf',
                  'GNN_Time','gap_time','TimeMIP','TimeGNN','speedup','instance']

# 按Alpha聚合
grouped = df_all.groupby('Alpha').agg(
    mean_gap=('gap_cmax', 'mean'),
    median_gap=('gap_cmax', 'median'),
    mean_time=('GNN_Time', 'mean'),
    mean_time_gap=('gap_time', 'mean'),
    mean_nsf = ('GNN_NSF','mean'),
    mean_nsf_ratio=('gap_nsf', 'mean')  # GNN_NSF/MIP_NSF 均值
).reset_index()


# 按Alpha降序排列（对应原图0.9→0.2）
grouped = grouped.sort_values('Alpha', ascending=False)

with pd.ExcelWriter(f"{filepath}alpha_analysis_summary.xlsx") as writer:
    df_all.to_excel(writer, sheet_name='raw_data', index=False)
    grouped.to_excel(writer, sheet_name='summary', index=False)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()  # 右轴：runtime

# 控制Alpha步长
plot_step = 0.1
plot_alphas = [round(i * plot_step, 2) for i in range(1, int(1/plot_step))]
plot_alphas = sorted(set([0.02] + plot_alphas))  # 添加0.02并去重排序

# 筛选grouped中对应的行
grouped_plot = grouped[grouped['Alpha'].round(2).isin(plot_alphas)]
grouped_plot = grouped_plot.sort_values('Alpha', ascending=True)


x = np.arange(len(grouped_plot))
labels = grouped_plot['Alpha'].round(2).astype(str).tolist()

# ---- 柱状图：GNN配置数占比 ----
bar_vals = grouped_plot['mean_nsf_ratio'] * 100
bars = ax1.bar(x, bar_vals, color='#f5d6a8', width=0.6, zorder=1, label='#NSF [%]')

# 柱顶标注数值
for xi, val in zip(x, bar_vals):
    ax1.text(xi, val + 0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='#a07040')

# ---- 左轴：gap折线 ----
ax1.plot(x, grouped_plot['mean_gap'],   'b-o', linewidth=2, markersize=5, label='mean gap',   zorder=3)
ax1.plot(x, grouped_plot['median_gap'], 'o-',  linewidth=2, markersize=5, label='median gap',
         color='orange', zorder=3)

# ---- 右轴：runtime折线 ----
ax2.plot(x, grouped_plot['mean_time'], 'g-o', linewidth=2, markersize=5, label='runtime', zorder=3)

# ---- 坐标轴装饰 ----
ax1.set_xlabel('Alpha (Probability threshold)', fontsize=11)
ax1.set_ylabel('Optimality gap [%]', fontsize=11)
ax2.set_ylabel('Runtime [s]', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

# ---- 图例合并 ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.title('Effect of Alpha on Solution Quality and Runtime')
plt.tight_layout()
plt.savefig(f"{filepath}alpha_analysis.png", dpi=300)
plt.savefig(f"{filepath}alpha_analysis.pdf", format='pdf', bbox_inches='tight')
plt.show()