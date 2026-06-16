# seru/utils/ccea/config_loader_old.py
import yaml
from pathlib import Path
from typing import Any


class ConfigLoader:
    def __init__(self, config_path: str):
        """
        支持加载任意位置的配置文件
        :param config_path: 支持绝对路径或相对于项目根目录的路径，如：
                   "seru/config/ccea.yaml"
                   "../config/experiment.yaml"
        """
        # 转换为绝对路径
        self.config_file = Path(__file__).parent.parent / config_path

        # 验证文件存在
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        # 加载配置
        self.config_data = self.load_config()

        # 替换 due_dates_path 中的占位符
        if 'due_dates_path' in self.config_data:
            self._replace_due_dates_path()

        # 动态绑定配置项（带冲突检查）
        self._bind_attributes()

    def load_config(self) -> dict:
        """ 加载YAML配置 """
        with open(self.config_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def _replace_due_dates_path(self):
        """ 替换 due_dates_path 中的占位符 """
        due_dates_path = self.config_data['due_dates_path']
        # 获取配置中的参数
        num_batches = self.config_data.get('num_of_batches', 0)
        num_workers = self.config_data.get('num_of_workers', 0)
        R = self.config_data.get('R', 0)
        T = self.config_data.get('T', 0)

        # 使用占位符替换
        formatted_path = due_dates_path.format(num_batches=num_batches, num_workers=num_workers, R=R, T=T)
        self.config_data['due_dates_path'] = formatted_path

    def _bind_attributes(self):
        """ 动态绑定属性，避免覆盖类方法 """
        reserved_names = {'config_data', 'config_file', 'load_config'}
        for key, value in self.config_data.items():
            if hasattr(self, key) or key in reserved_names:
                raise AttributeError(f"配置键 '{key}' 与类属性或保留名称冲突")
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """ 访问未定义属性时给出明确错误提示 """
        raise AttributeError(f"'{self.__class__.__name__}' 没有 '{name}' 属性，请检查配置文件: {self.config_file}")
