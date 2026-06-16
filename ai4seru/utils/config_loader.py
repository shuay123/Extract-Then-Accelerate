# seru/utils/ccea/config_loader_old.py
import yaml
from pathlib import Path
from typing import Any, Dict

# todo 后续全部转成 hydra 管理
class ConfigLoader:
    # 类级别的实例缓存
    _instances: Dict[str, 'ConfigLoader'] = {}
    # 预加载配置列表（按需添加）
    _preload_configs = [
        "config/config_seru.yaml",
        "config/config_ga.yaml",
        "config/config_ccea.yaml",
        "config/config_drl.yaml"
        # todo 已经训练过的模型的config直接再次使用，如何设计
        # "config/config_drl_trained_model"
        # 后续新增配置在此添加
    ]
    print("使用的配置文件如下")
    print(_preload_configs)

    @classmethod
    def get_config(cls, name: str) -> 'ConfigLoader':
        """通过配置名称获取实例（需要预配置映射）"""
        config_map = {
            "config_seru": "config/config_seru.yaml",
            "config_ga": "config/config_ga.yaml",
            "config_ccea": "config/config_ccea.yaml",
            "config_drl": "config/config_drl.yaml"
            # "config_drl": "config/config_drl_trained_model.yaml"
        }
        return cls(config_map[name.lower()])

    def __new__(cls, config_path: str):
        """支持单例和多配置管理的初始化"""
        # 转换为绝对路径
        resolved_path = (Path(__file__).parent.parent / config_path).resolve()

        # 路径标准化处理
        canonical_path = str(resolved_path)

        # 检查是否已存在实例
        if canonical_path not in cls._instances:
            # 文件存在性验证
            if not resolved_path.exists():
                raise FileNotFoundError(f"Config file not found: {resolved_path}")

            # 创建新实例
            instance = super().__new__(cls)
            instance._initialize(resolved_path)
            cls._instances[canonical_path] = instance

        return cls._instances[canonical_path]

    def _initialize(self, config_path: Path):
        """实际的初始化逻辑"""
        self.config_file = config_path
        self.config_data = self._load_config()

        # 路径替换处理
        self._process_path_placeholders()

        # 动态绑定属性
        self._bind_attributes()

    @classmethod
    def preload_all(cls):
        """项目启动时预加载所有配置"""
        for path in cls._preload_configs:
            try:
                cls(path)
            except Exception as e:
                print(f"预加载配置 {path} 失败: {e}")

    def _load_config(self) -> dict:
        """加载YAML文件"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _process_path_placeholders(self):
        """处理所有路径占位符"""
        # due_dates_path 处理
        if 'due_dates_path' in self.config_data:
            params = {
                'num_batches': self.config_data.get('num_of_batches', 0),
                'num_workers': self.config_data.get('num_of_workers', 0),
                'R': self.config_data.get('R', 0),
                'T': self.config_data.get('T', 0)
            }
            self.config_data['due_dates_path'] = self.config_data['due_dates_path'].format(**params)
        # 其他路径处理可以在此扩展
        if 'batch_types_path' in self.config_data:
            params = {
                'num_of_batch_types': self.config_data.get('num_of_batch_types', 0)
            }
            self.config_data['batch_types_path'] = self.config_data['batch_types_path'].format(**params)

    def _bind_attributes(self):
        """安全绑定配置项为属性"""
        reserved = {'config_data', 'config_file'} | set(dir(self))
        for key, value in self.config_data.items():
            if key in reserved:
                raise AttributeError(f"配置键 '{key}' 冲突")
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """增强错误提示"""
        raise AttributeError(f"配置 '{self.config_file.name}' 中不存在属性: {name}")
