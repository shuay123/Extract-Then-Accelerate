# utils/excel_output.py
import os
# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import xlwt

class ExcelOutputUtils:
    """Excel输出工具类"""

    @staticmethod
    def save_ranking_data_to_excel(ranking_data: List[Dict], num_workers: int, num_batches: int, excel_path: str = "data/result"):
        """
        将排位数据保存到Excel文件

        Args:
            ranking_data: 包含排位信息的字典列表
            num_workers: 工人数量
            num_batches: 批次数量
            excel_path: Excel文件路径（相对于项目根目录）
        """
        excel_full_path = Path(__file__).parent.parent / excel_path
        excel_full_path.mkdir(parents=True, exist_ok=True)

        # 创建DataFrame
        df = pd.DataFrame(ranking_data)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_ranking_analysis_w{num_workers}_b{num_batches}_{timestamp}.xlsx"
        output_file = excel_full_path / filename

        # 保存到Excel
        df.to_excel(output_file, index=False, sheet_name="批次排位分析")

        print(f"排位数据已保存到: {output_file}")
        return output_file

    @staticmethod
    def plot_ranking_analysis(ranking_data: List[Dict], num_workers: int, num_batches: int):
        """
        绘制三张分析图表

        Args:
            ranking_data: 包含排位信息的字典列表
            num_workers: 工人数量
            num_batches: 批次数量
        """
        if not ranking_data:
            print("没有数据可以绘制图表")
            return

        # 转换为DataFrame方便处理
        df = pd.DataFrame(ranking_data)

        # 设置中文字体（可选，如果需要显示中文）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'批次调度分析 (工人数: {num_workers}, 批次数: {num_batches})', fontsize=16, fontweight='bold')

        # 图1: 归一化排位趋势
        ax1 = axes[0]
        ax1.plot(df['step'], df['normalized_slack_rank'], 'b-', label='Normalized Slack Rank', linewidth=1.5)
        ax1.plot(df['step'], df['normalized_fitness_rank'], 'r-', label='Normalized Fitness Rank', linewidth=1.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Normalized Rank')
        ax1.set_title('归一化排位趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 图2: Slack Time 值
        ax2 = axes[1]
        ax2.plot(df['step'], df['selected_batch_slack_time'], 'g-', label='Selected Batch Slack Time', linewidth=1.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Slack Time')
        ax2.set_title('被选批次的Slack Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3: Fitness 值
        ax3 = axes[2]
        ax3.plot(df['step'], df['selected_batch_fitness'], 'm-', label='Selected Batch Fitness', linewidth=1.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Fitness')
        ax3.set_title('被选批次的Fitness')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 显示图表
        plt.show()

        # 打印一些统计信息
        print("\n=== 统计信息 ===")
        print(f"平均归一化Slack排位: {df['normalized_slack_rank'].mean():.3f}")
        print(f"平均归一化Fitness排位: {df['normalized_fitness_rank'].mean():.3f}")
        print(f"平均Slack Time: {df['selected_batch_slack_time'].mean():.2f}")
        print(f"平均Fitness: {df['selected_batch_fitness'].mean():.3f}")
        print(f"Slack Time标准差: {df['selected_batch_slack_time'].std():.2f}")
        print(f"Fitness标准差: {df['selected_batch_fitness'].std():.3f}")

    def save_serudata_to_excel(seru_data: List[Dict], num_workers: int, num_batches: int, excel_path: str = "data/result"):
        """
        将服务数据保存到Excel文件

        Args:
            seru_data: 包含服务信息的字典列表
            num_workers: 工人数量
            num_batches: 批次数量
            excel_path: Excel文件路径（相对于项目根目录）
        """
        excel_full_path = Path(__file__).parent.parent / excel_path
        excel_full_path.mkdir(parents=True, exist_ok=True)

        # 创建DataFrame
        df = pd.DataFrame(seru_data)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"seru_analysis_w{num_workers}_b{num_batches}_{timestamp}.xlsx"
        output_file = excel_full_path / filename

        # 保存到Excel
        df.to_excel(output_file, index=False, sheet_name="服务分析")

def save_serudata_to_excel( filename: str, sheets):
    """
    将服务数据保存到Excel文件

    Args:
        filename: 输出文件名（不包含扩展名）
        sheets: 包含工作表名称和数据的列表，每个元素为 {'SheetName': str, 'Data': List[List]}
        每个Data元素为二维列表，第一行为表头，后续行为数据    
    """
    # 获取文件路径中的目录部分
    directory = os.path.dirname(filename)
    
    # 如果目录路径不为空，且该目录不存在，则创建它
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"创建目录失败: {e}")
    # 创建工作簿和工作表
    workbook = xlwt.Workbook()
    for sheet in sheets:
        sheet_name = sheet['SheetName']
        data = sheet['Data']
        save_data_to_book(workbook, sheet_name, data)

    # 保存为 .xls 文件
    full_path = filename + '.xls'
    workbook.save(full_path)
    

def save_data_to_book(workbook: xlwt.Workbook, sheet_name: str, data: List[List]):
    """
    将数据保存到Excel工作簿的指定工作表

    Args:
        workbook: xlwt.Workbook对象
        sheet_name: 工作表名称
        data: 二维列表，包含要写入的数据
    """
    worksheet = workbook.add_sheet(sheet_name)
    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            worksheet.write(row_idx, col_idx, value)

if __name__ == "__main__":
    sheets_names = ['dasd', 'das12d']
    data_list =  [
        [['姓名a', '数s学', '英语f'], ['sdAlice', 9120, 8315], ['asdBob', 80, 88]],
        [['姓名', '数学', '英语'], ['Alice', 20, 15], ['Bob', 80, 88]]
    ]
    sheets = []
    for sheet_name, data in zip(sheets_names, data_list):
        sheets.append({'SheetName': sheet_name, 'Data': data})
    save_serudata_to_excel('test', sheets_names, data_list,sheets)
