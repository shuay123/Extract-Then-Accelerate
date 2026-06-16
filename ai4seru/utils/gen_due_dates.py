import os
import pandas as pd
import random
from pathlib import Path
from config_loader import ConfigLoader
from excel_utils import ExcelDataLoader
from problem.pure_seru.calculate_fitness import CalculateFitness
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule
from problem.pure_seru.initialization import Initialization


class GenDueDate:
    def __init__(self):
        self.config_seru = ConfigLoader.get_config('config_seru')
        self.excel_loader = ExcelDataLoader()
        self.excel_loader.read_data(excel_path=self.config_seru.seru_data_path, config_sheet=self.config_seru)
        self.excel_loader.read_data(excel_path=self.config_seru.batch_types_path, config_sheet=self.config_seru)
        formation_code = list(range(1, self.config_seru.num_of_workers + 1))
        self.formation = SeruFormation(formation_code=formation_code)
        Initialization.produce_seru_formation(self.config_seru.num_of_workers, self.formation)

        schedule_code = list(range(1, self.config_seru.num_of_batches + 1))
        self.schedule = SeruSchedule(schedule_code=schedule_code)
        Initialization.produce_seru_schedule(self.config_seru.num_of_batches, self.schedule)

    def calculate_total_throughput_time(self):
        seru_schedule = self.schedule.batches_assignment
        batches = seru_schedule[0]
        seru = self.formation.seru_set[0]

        for batch_id in batches:
            # 将批次添加到 Seru 单元
            seru.batches_set.append(batch_id)

            # 计算单个batch的throughput
            throughput_time_of_batch = CalculateFitness.calculate_batch_throughput_time(batch_id=batch_id, config_seru= self.config_seru, excel_loader=self.excel_loader, seru=seru)


            # 更新 Seru 单元的总流动时间，不考虑换装时间
            seru.throughput_time += throughput_time_of_batch

            # 记录批次的流动时间
            self.schedule.batches_throughput_time_in_seru.append((batch_id, seru.throughput_time))
        self.formation.makespan = round(max(seru.throughput_time for seru in self.formation.seru_set), 3)

    def generate_due_dates(self, P: float) -> list:
        lower_bound = P * (1 - self.config_seru.T - self.config_seru.R / 2)
        upper_bound = P * (1 - self.config_seru.T + self.config_seru.R / 2)
        due_dates = [
            random.randint(int(lower_bound), int(upper_bound))
            for _ in range(self.config_seru.num_of_batches)
        ]
        due_dates.sort()
        return due_dates

    def write_due_dates_to_excel(self, due_dates):
        standard_suffix = "_standard" if getattr(self.config_seru, 'use_standard_logic', False) else ""

        filename = (
            f"due_dates_b{self.config_seru.num_of_batches}"
            f"_w{self.config_seru.num_of_workers}"
            f"_R{self.config_seru.R}"
        f"_T{self.config_seru.T}{standard_suffix}.xlsx"
        )

        target_dir = Path(__file__).parent.parent / "data/due_date"
        os.makedirs(target_dir, exist_ok=True)
        filepath = os.path.join(target_dir, filename)

        df = pd.DataFrame({
            "批次": range(1, len(due_dates) + 1),
            "批次截止时间": due_dates
        })
        df.to_excel(filepath, index=False, sheet_name='均匀分布生成的截止时间')
        print(f"文件已保存至：{filepath}")


if __name__ == "__main__":
    ConfigLoader.preload_all()
    gen_due_date = GenDueDate()
    gen_due_date.calculate_total_throughput_time()
    P = gen_due_date.formation.makespan

    due_dates = gen_due_date.generate_due_dates(P)
    gen_due_date.write_due_dates_to_excel(due_dates=due_dates)

    print(f"\n总处理时间: {P}")
    print(f"生成参数: R={gen_due_date.config_seru.R}, T={gen_due_date.config_seru.T}")
    print(f"截止日期范围: [{min(due_dates)}, {max(due_dates)}]")
