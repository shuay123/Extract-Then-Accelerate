import yaml
import os

def load_config(config_path):
    """
    加载YAML配置文件。
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误：配置文件未找到！请检查路径：{config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"错误：解析YAML文件时出错：{e}")
        return None

# 1. 获取当前Python脚本所在的目录 (例如: my_project/util/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 回溯到父目录 (例如: my_project/)
# os.path.dirname() 再调用一次，就能从当前目录向上回溯一级
parent_dir = os.path.dirname(current_script_dir)

# 3. 构建 config 文件夹的路径 (例如: my_project/config/)
config_folder_path = os.path.join(parent_dir, 'config')

# 4. 构建 config.yaml 文件的完整路径 (例如: my_project/config/config.yaml)
config_file_path = os.path.join(config_folder_path, 'config.yaml')

print(f"尝试加载的配置文件路径: {config_file_path}")

def load_yaml_config(yaml_file_name):
    """
    加载YAML配置文件。
    """
    # 1. 获取当前Python脚本所在的目录 (例如: my_project/util/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 回溯到父目录 (例如: my_project/)
    # os.path.dirname() 再调用一次，就能从当前目录向上回溯一级
    parent_dir = os.path.dirname(current_script_dir)

    # 3. 构建 config 文件夹的路径 (例如: my_project/config/)
    config_folder_path = os.path.join(parent_dir, 'config')

    # 4. 构建 config.yaml 文件的完整路径 (例如: my_project/config/config.yaml)
    config_file_path = os.path.join(config_folder_path, yaml_file_name)

    print(f"尝试加载的配置文件路径: {config_file_path}")
    params = load_config(config_file_path)
    return params

# 加载配置

if __name__ == "__main__":
    params = load_yaml_config('JCompany_W6_J6_5000.yaml')
    if params:
        print("\n--- 成功加载配置 ---")
        print(f"数据文件路径 (data_dir): {params.get('data_dir', '未设置')}")
        print(f"节点数量 (n_nodes): {params.get('n_nodes', '未设置')}")
        print(f"批处理大小 (batch_size): {params.get('batch_size', '未设置')}")
        print(f"学习率 (lr): {params.get('lr', '未设置')}")
        print(f"是否使用对比学习 (use_contrastive): {params.get('use_contrastive', '未设置')}")

        # 示例：将参数传递给一个函数
        def train_model(data_dir, n_nodes, batch_size, lr, **kwargs):
            print("\n--- 开始训练模型 ---")
            print(f"使用数据: {data_dir}")
            print(f"节点数: {n_nodes}")
            print(f"批大小: {batch_size}")
            print(f"学习率: {lr}")
            print(f"其他参数: {kwargs}")
            print("--- 训练完成 ---")

        train_model(**params)
    else:
        print("\n配置加载失败，程序无法继续。")




