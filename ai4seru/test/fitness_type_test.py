from utils.excel_utils import ExcelDataLoader
from utils.config_loader import ConfigLoader
from problem.pure_seru.pure_seru_entities import Seru, SeruSchedule, SeruFormation, Solution
from problem.pure_seru.calculate_fitness import CalculateFitness
from problem.pure_seru.initialization import Initialization

ConfigLoader.preload_all()
loader = ExcelDataLoader()
config_seru = ConfigLoader.get_config('config_seru')
loader.read_data(excel_path=config_seru.seru_data_path, config_sheet=config_seru)
loader.read_data(excel_path=config_seru.due_dates_path, config_sheet=config_seru)
loader.read_data(excel_path=config_seru.batch_types_path, config_sheet=config_seru)

# 初始化 Seru, SeruSchedule 和 SeruFormation 类
seru = Seru()
schedule = SeruSchedule()
formation = SeruFormation()

# 查看它们的初始状态
print(seru)
print(schedule)
print(formation)

# 初始化 formation_code
num_of_workers = 5
# formation.formation_code = Initialization.initial_formation_code(num_of_workers)
formation.formation_code = [1,2,3,4,5]

# 查看初始化后的 formation_code
# print("Formation Code:", formation.formation_code)
# 初始化 Seru Formation
Initialization.produce_seru_formation(num_of_workers, formation)

for i, seru_instance in enumerate(formation.seru_set):
    print(f"Seru {i + 1}: {seru_instance.workers_set}")
schedule.schedule_code = list(range(1, config_seru.num_of_batches + 1))

# 查看生成的 schedule_code
# print("Schedule Code:", schedule.schedule_code)
# 初始化 Seru Schedule
Initialization.produce_seru_schedule(config_seru.num_of_batches, schedule)

# 查看生成的 batches_assignment
# print("Batches Assignment:")
# for batch in schedule.batches_assignment:
#     print(batch)
# 创建 Solution 实例
solution = Solution(formation=formation, schedule=schedule)
is_tardiness = False
# 计算适应度
CalculateFitness.calculate_fitness(solution, config_seru)

# 查看计算结果
print("Total Makespan:", solution.makespan)
print("Total Tardiness:", solution.tardiness)
# print("Total Labour Time:", solution.labour_time)


# 查看每个 Seru 单元的流动时间和劳动时间
# for i, seru_instance in enumerate(formation.seru_set):
#     print(f"Seru {i + 1}:")
#     print(f"  Makespan: {seru_instance.throughput_time}")
#     print(f"  Labour Time: {seru_instance.labour_time}")
