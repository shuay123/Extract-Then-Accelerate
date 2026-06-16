from utils.config_loader import ConfigLoader

if __name__ == "__main__":
    config = ConfigLoader.get_config("config_seru")
    config_dict = config.config_data  # 直接获取字典
    print(config_dict)
