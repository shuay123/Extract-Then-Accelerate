# 在项目入口文件（如main.py）最前面添加
from utils.config_loader import ConfigLoader
import test1

ConfigLoader.preload_all()  # 预加载所有配置
test1.test()
