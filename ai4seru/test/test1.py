from utils.config_loader import ConfigLoader


def test():
    config_seru = ConfigLoader.get_config('config_seru')
    print(config_seru.NUM_OF_WORKERS)
