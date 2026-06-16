# metaheuristics/ccea/ccea.py
import sys
import os
# 添加项目根目录到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import time
from typing import List
from utils.excel_utils import ExcelDataLoader
from metaheuristics.common.ga_operator import GaOperator

# 纯seru生产则使用下面的导入
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.initialization import Initialization
from problem.pure_seru.calculate_fitness import CalculateFitness


class CCEA:
    def __init__(self, worker_map=None, batch_map=None, PSF=None, PSS=None):
        """
        初始化协同进化算法
        """
        # 加载配置
        self.config_seru = ConfigLoader.get_config('config_seru')
        self.config_ccea = ConfigLoader.get_config('config_ccea')

        self.PSF = PSF
        self.PSS = PSS

        self.config_seru.worker_map = worker_map
        self.config_seru.batch_map = batch_map
        self.config_seru.selected_batches = list(batch_map.values())
        # 初始化 Seru 数据
        self.loader = ExcelDataLoader()
        self.loader.read_data(excel_path=self.config_seru.seru_data_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.due_dates_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.batch_types_path, config_sheet=self.config_seru)

    def run(self) -> (SeruFormation, SeruSchedule, Solution):
        """
        运行协同进化算法
        :return: 最佳 Seru 构造、最佳 Seru 调度、最佳解
        """
        # ------------------------------
        # 1. 初始化种群
        # ------------------------------
        # 初始化 Seru 构造种群
        PSF = self.PSF
        PSS = self.PSS

        # ------------------------------
        # 2. 初始化最佳解
        # ------------------------------
        best_formation: SeruFormation = PSF[0]  # 显式指定 best_formation 是一个 SeruFormation 类型
        best_schedule: SeruSchedule = PSS[0]  # 显式指定 best_scheduling 是一个 SeruSchedule 类型

        # 创建当前解
        current_best_solution: Solution = Solution(formation=best_formation, schedule=best_schedule)

        # 计算适应度
        CalculateFitness.calculate_fitness(solution=current_best_solution, config_seru=self.config_seru)

        # 初始化最佳解
        best_solution: Solution = current_best_solution  # 显式指定 best_solution 是一个 Solution 类型

        # ------------------------------
        # 3. 主循环
        # ------------------------------
        start_time = time.time()  # 记录算法开始时间
        iteration = 0  # 迭代计数器
        while iteration < self.config_ccea.max_iteration:
            iteration += 1  # 更新迭代计数器
            # ------------------------------
            # 3.1 进化 Seru 构造种群
            # ------------------------------
            # 计算适应度
            for formation in PSF:  # PSF: List[SeruFormation]
                solution: Solution = Solution(formation=formation, schedule=best_schedule)  # Solution 类型
                CalculateFitness.calculate_fitness(solution, self.config_seru)

            # 选择操作，保持种群大小不变
            selected: List[SeruFormation] = [GaOperator.tournament_selection(PSF) for _ in
                                             range(len(PSF))]  # List[SeruFormation]

            # 交叉和变异
            offspring: List[SeruFormation] = []  # ` List[SeruFormation]
            for i in range(0, len(selected), 2):
                parent_formation1, parent_formation2 = selected[i], selected[i + 1]

                # 先进行深拷贝
                child_formation1, child_formation2 = parent_formation1.__copy__(), parent_formation2.__copy__()

                # 进行交叉操作
                child_formation1.formation_code, child_formation2.formation_code = GaOperator.order_crossover(
                    parent_formation1.formation_code,
                    parent_formation2.formation_code,
                    self.config_ccea)

                # 进行变异操作
                child_formation1.formation_code = GaOperator.swap_mutation(child_formation1.formation_code,
                                                                           self.config_ccea)
                child_formation2.formation_code = GaOperator.swap_mutation(child_formation2.formation_code,
                                                                           self.config_ccea)

                Initialization.produce_seru_formation(self.config_seru.num_of_workers, child_formation1)
                Initialization.produce_seru_formation(self.config_seru.num_of_workers, child_formation2)

                # 将变异后的子代添加到 offspring 列表
                offspring.append(child_formation1)
                offspring.append(child_formation2)

            # 更新种群和最佳构造
            PSF = offspring  # List[SeruFormation]
            for formation in PSF:
                solution: Solution = Solution(formation=formation, schedule=best_schedule)
                CalculateFitness.calculate_fitness(solution, self.config_seru)
            best_formation: SeruFormation = min(PSF, key=lambda x: x.fitness)  # SeruFormation 类型

            # 局部搜索：优化最佳构造
            best_formation = GaOperator.local_search_formation(formation=best_formation, best_scheduling=best_schedule, config_ccea=self.config_ccea, config_seru=self.config_seru)

            # ------------------------------
            # 3.2 进化 Seru 调度种群
            # ------------------------------
            # 计算适应度
            for schedule in PSS:  # PSS: List[SeruSchedule]
                solution: Solution = Solution(formation=best_formation, schedule=schedule)  # Solution 类型
                CalculateFitness.calculate_fitness(solution, self.config_seru)

            # 选择操作，保持种群大小不变
            selected: List[SeruSchedule] = [GaOperator.tournament_selection(PSS) for _ in range(len(PSS))]

            # 交叉和变异
            offspring: List[SeruSchedule] = []  # List[SeruSchedule]
            for i in range(0, len(selected), 2):
                parent_schedule1, parent_schedule2 = selected[i], selected[i + 1]

                # 先进行深拷贝
                child_schedule1, child_schedule2 = parent_schedule1.__copy__(), parent_schedule2.__copy__()

                # 进行交叉操作
                child_schedule1.schedule_code, child_schedule2.schedule_code = GaOperator.order_crossover(
                    parent_schedule1.schedule_code,
                    parent_schedule2.schedule_code,
                    self.config_ccea)

                # 进行变异操作
                child_schedule1.schedule_code = GaOperator.swap_mutation(child_schedule1.schedule_code,
                                                                         self.config_ccea)
                child_schedule2.schedule_code = GaOperator.swap_mutation(child_schedule2.schedule_code,
                                                                         self.config_ccea)

                Initialization.produce_seru_schedule(self.config_seru.num_of_batches, child_schedule1)
                Initialization.produce_seru_schedule(self.config_seru.num_of_batches, child_schedule2)

                # 将变异后的子代添加到 offspring 列表
                offspring.append(child_schedule1)
                offspring.append(child_schedule2)

            # 更新种群和最佳调度
            PSS = offspring  # List[SeruSchedule]
            for schedule in PSS:
                solution: Solution = Solution(formation=best_formation, schedule=schedule)
                CalculateFitness.calculate_fitness(solution, self.config_seru)
            best_schedule: SeruSchedule = min(PSS, key=lambda x: x.fitness)  # SeruSchedule 类型

            # 局部搜索：优化最佳调度
            best_schedule = GaOperator.local_search_schedule(schedule=best_schedule, best_formation=best_formation, config_seru=self.config_seru, config_ccea=self.config_ccea)

            current_best_solution = Solution(formation=best_formation, schedule=best_schedule)
            CalculateFitness.calculate_fitness(current_best_solution, self.config_seru)

            # ------------------------------
            # 3.3 更新当前解
            # ------------------------------
            if current_best_solution.fitness < best_solution.fitness:
                best_solution: Solution = current_best_solution  # Solution 类型

                # for i in range(len(best_solution.formation.seru_set)):
                #     print("Seru %d: %s" % (i + 1, len(best_solution.formation.seru_set[i].workers_set)))

            # 每 100 次迭代输出一次最优解的 fitness 和代数
            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time  # 计算已用时间
                print(f"迭代次数: {iteration}, 适应值: {best_solution.fitness}，耗时: {elapsed_time:.2f}秒")
                # 使用 enumerate 获取序号，并用列表推导式生成格式化的字符串
                seru_info = [f"Seru {i}: {len(seru.workers_set)}" for i, seru in enumerate(best_solution.formation.seru_set, start=1)]
                # print("Seru数量:", ", ".join(seru_info))

        # ------------------------------
        # 4. 返回结果
        # ------------------------------
        return best_solution.formation, best_solution.schedule, best_solution


from heuristics.formation_heuristic import FormationHeuristic
from utils.config_loader import ConfigLoader
import random

def main():
    # 预加载配置
    ConfigLoader.preload_all()
    config_seru = ConfigLoader.get_config('config_seru')

    # === [修正 3]：生成随机映射 ===
    # 假设 Excel 中有 1-40 号工人，1-500 号批次
    all_real_workers = list(range(1, 41))
    all_real_batches = list(range(1, 501))

    selected_workers = random.sample(all_real_workers, config_seru.num_of_workers)
    selected_batches = random.sample(all_real_batches, config_seru.num_of_batches)

    worker_map = {
        logic_id + 1: real_id
        for logic_id, real_id in enumerate(selected_workers)
    }
    
    batch_map = {
        logic_id + 1: real_id
        for logic_id, real_id in enumerate(selected_batches)
    }

    print(f"随机映射工人: {worker_map}")
    print(f"随机映射批次: {batch_map}")
    # ==========================

    # 传入映射表
    ccea = CCEA(worker_map, batch_map)

    # 运行协同进化算法
    best_formation, best_scheduling, best_solution = ccea.run()

    # FormationHeuristic.formation_heuristic(best_solution)

    # 输出结果
    print("Best Formation:", best_formation)
    for i in range(len(best_formation.seru_set)):
        print("Seru %d: %s" % (i + 1, len(best_formation.seru_set[i].workers_set)))
    print("Best Scheduling:", best_scheduling)
    print("Best Solution:", best_solution)



# 确保只有在直接运行此文件时才会启动
if __name__ == "__main__":
    main()
