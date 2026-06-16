import os
import pandas as pd
from pathlib import Path


class ExcelDataLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # 避免重复初始化
            self.initialized = True

            # 添加字典存储
            self.batch_to_product_dict = None
            self.worker_to_product_dict = None
            self.worker_to_task_dict = None
            self.batch_due_dates_dict = None

    def read_data(self, excel_path, config_sheet):
        try:
            excel_full_path = Path(__file__).parent.parent / excel_path
            excel_file = pd.ExcelFile(excel_full_path)

            for sheet_name in excel_file.sheet_names:
                sheet = excel_file.parse(sheet_name)
                if sheet_name == config_sheet.sheet_batch_to_product:
                    self.batch_to_product_dict = sheet.set_index('批次').to_dict('index')
                    # print(self.batch_to_product_dict)
                elif sheet_name == config_sheet.sheet_worker_to_product:
                    self.worker_to_product_dict = sheet.set_index('工人、产品类型').to_dict('index')
                    # print(self.worker_to_product_dict)
                elif sheet_name == config_sheet.sheet_worker_to_multiple_task:
                    self.worker_to_task_dict = sheet.set_index('工人').to_dict('index')
                    # print(self.worker_to_task_dict)
                elif sheet_name == config_sheet.sheet_batch_due_dates:
                    self.batch_due_dates_dict = sheet.set_index('批次').to_dict('index')
                    # print(self.batch_due_dates_dict)
        except FileNotFoundError:
            print(f"File not found: {excel_path}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")

    # 🟢 [新增] 
    def load_data_from_dicts(self, worker_task_data, worker_prod_data, batch_prod_data, batch_due_dates_data=None):
        """
        直接从字典加载数据，而不是从 Excel 文件。
        """
        # 规范化 worker_to_task_dict：键为 int，包含 '系数' 且为数值
        normalized_worker_task = {}
        for k, v in (worker_task_data or {}).items():
            try:
                key = int(k)
            except Exception:
                raise ValueError(f"worker_to_task_dict 的键不可转换为整数: {k}")
            if '系数' not in v:
                raise ValueError(f"worker_to_task_dict[{key}] 缺少必需字段 '系数'")
            try:
                coeff = float(v['系数'])
            except Exception:
                raise ValueError(f"worker_to_task_dict[{key}]['系数'] 必须为数值")
            normalized_worker_task[key] = {'系数': coeff}

        # 规范化 worker_to_product_dict：外层键为 int，内层产品类型键为 int，值为数值
        normalized_worker_prod = {}
        for wk, mapping in (worker_prod_data or {}).items():
            try:
                wk_id = int(wk)
            except Exception:
                raise ValueError(f"worker_to_product_dict 的键不可转换为整数: {wk}")
            if not isinstance(mapping, dict):
                raise ValueError(f"worker_to_product_dict[{wk_id}] 必须为字典")
            inner = {}
            for pt, val in mapping.items():
                try:
                    pt_id = int(pt)
                except Exception:
                    raise ValueError(f"worker_to_product_dict[{wk_id}] 的产品类型键不可转换为整数: {pt}")
                try:
                    coeff = float(val)
                except Exception:
                    raise ValueError(f"worker_to_product_dict[{wk_id}][{pt_id}] 的值必须为数值")
                inner[pt_id] = coeff
            normalized_worker_prod[wk_id] = inner

        # 规范化 batch_to_product_dict：键为 int，包含 '产品类型' 与 '批次大小'，两者均为数值
        normalized_batch_prod = {}
        for bk, v in (batch_prod_data or {}).items():
            try:
                b_id = int(bk)
            except Exception:
                raise ValueError(f"batch_to_product_dict 的键不可转换为整数: {bk}")
            if not isinstance(v, dict):
                raise ValueError(f"batch_to_product_dict[{b_id}] 必须为字典")
            if '产品类型' not in v or '批次大小' not in v:
                raise ValueError(f"batch_to_product_dict[{b_id}] 缺少必需字段 '产品类型' 或 '批次大小'")
            try:
                prod_type = int(v['产品类型'])
            except Exception:
                raise ValueError(f"batch_to_product_dict[{b_id}]['产品类型'] 必须为整数")
            try:
                batch_size = int(v['批次大小'])
            except Exception:
                raise ValueError(f"batch_to_product_dict[{b_id}]['批次大小'] 必须为整数")
            normalized_batch_prod[b_id] = {'产品类型': prod_type, '批次大小': batch_size}

        # 规范化 batch_due_dates_dict（可选）：键为 int，包含 '批次截止时间' 为数值
        normalized_due_dates = {}
        for dk, v in (batch_due_dates_data or {}).items():
            try:
                d_id = int(dk)
            except Exception:
                raise ValueError(f"batch_due_dates_dict 的键不可转换为整数: {dk}")
            if not isinstance(v, dict) or '批次截止时间' not in v:
                raise ValueError(f"batch_due_dates_dict[{d_id}] 缺少必需字段 '批次截止时间'")
            try:
                due = float(v['批次截止时间'])
            except Exception:
                raise ValueError(f"batch_due_dates_dict[{d_id}]['批次截止时间'] 必须为数值")
            normalized_due_dates[d_id] = {'批次截止时间': due}

        self.worker_to_task_dict = normalized_worker_task
        self.worker_to_product_dict = normalized_worker_prod
        self.batch_to_product_dict = normalized_batch_prod
        self.batch_due_dates_dict = normalized_due_dates
        print("ExcelDataLoader 已从传入的字典中加载数据。")

        
    @classmethod
    def instance(cls):
        """获取单例实例"""
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
