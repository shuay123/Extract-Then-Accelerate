import random
from pathlib import Path
import os
import pandas as pd
from config_loader import ConfigLoader


class GenBatchTypes:
    def __init__(self):
        self.config_seru = ConfigLoader.get_config('config_seru')

    def generate_batch_types(self):
        """
        生成批次类型字典：
        1. 从配置中读取批次类型数量和批次数量。
        2. 为每个批次生成随机类型。
        :return: 批次类型字典
        """
        num_of_batch_types = self.config_seru.num_of_batch_types
        num_of_batches = self.config_seru.num_of_batches
        batch_types = {
            i: random.randint(1, num_of_batch_types)
            for i in range(1, num_of_batches + 1)
        }
        return batch_types

    def write_batch_types_to_excel(self):
        """
        将批次类型写入Excel文件：
        1. 生成文件名并创建目标目录。
        2. 使用DataFrame保存数据。
        """
        batch_types = self.generate_batch_types()
        config = self.config_seru
        filename = (
            f"batch_types_{config.num_of_batch_types}.xlsx"
        )
        target_dir = Path(__file__).parent.parent / "data/batch_type"
        os.makedirs(target_dir, exist_ok=True)
        filepath = target_dir / filename

        df = pd.DataFrame({
            "批次": batch_types.keys(),
            "产品类型": batch_types.values(),
            "批次大小": [1] * len(batch_types)
        })
        df.to_excel(filepath, index=False, sheet_name='批次与产品类型关系_京东')
        print(f"文件已保存至：{filepath}")


if __name__ == "__main__":
    ConfigLoader.preload_all()
    gen_batch_types = GenBatchTypes()
    gen_batch_types.write_batch_types_to_excel()
