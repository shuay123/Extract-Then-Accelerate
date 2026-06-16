import time
from typing import List
from metaheuristics.common.ga_operator import GaOperator
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.initialization import Initialization
from problem.pure_seru.calculate_fitness import CalculateFitness
from utils.config_loader import ConfigLoader
from utils.excel_utils import ExcelDataLoader


class GA:
    def __init__(self):
        """
        初始化协同进化算法
        """
        # 加载配置
        self.config_seru = ConfigLoader.get_config('config_seru')
        self.config_ga = ConfigLoader.get_config('config_ga')

        # 初始化 Seru 数据
        self.loader = ExcelDataLoader()
        self.loader.read_data(excel_path=self.config_seru.seru_data_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.due_dates_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.batch_types_path, config_sheet=self.config_seru)

    def run(self, ) -> (SeruFormation, SeruSchedule, Solution):
        """
        运行协同进化算法
        :return: 最佳 Seru 构造、最佳 Seru 调度、最佳解
        """
        # ------------------------------
        # 1. 初始化种群
        # ------------------------------
        # 验证 formation_code 有效性
        num_workers = self.config_seru.num_of_workers
        required_workers = set(range(1, num_workers + 1))
        formation_code_set = set(self.config_ga.formation_code)

        if not required_workers.issubset(formation_code_set):
            raise ValueError("formation_code和num_of_workers不匹配")

        # 初始化 Seru 构造
        formation = SeruFormation(formation_code=self.config_ga.formation_code)
        Initialization.produce_seru_formation(self.config_seru.num_of_workers, formation)

        # 初始化 Seru 调度种群
        PSS: List[SeruSchedule] = []  # 显式指定 PSS 是一个 SeruSchedule 类型的列表
        for _ in range(self.config_ga.population_size):
            # schedule_code = list(range(1, self.config_seru.num_of_batches + 1))
            schedule_code = Initialization.initial_schedule_code(self.config_seru.num_of_workers, self.config_seru.num_of_batches)
            schedule = SeruSchedule(schedule_code=schedule_code)
            Initialization.produce_seru_schedule(self.config_seru.num_of_batches, schedule)
            PSS.append(schedule)

        best_solution: Solution = Solution(formation=formation, schedule=PSS[0])
        CalculateFitness.calculate_fitness(solution=best_solution, config_seru=self.config_seru)

        # ------------------------------
        # 3. 主循环
        # ------------------------------
        start_time = time.time()  # 记录算法开始时间
        iteration = 0  # 迭代计数器
        while time.time() - start_time < self.config_ga.max_runtime:
            iteration += 1  # 更新迭代计数器
            # ------------------------------
            # 3.2 进化 Seru 调度种群
            # ------------------------------
            # 计算适应度
            for schedule in PSS:  # PSS: List[SeruSchedule]
                solution: Solution = Solution(formation=formation, schedule=schedule)  # Solution 类型
                CalculateFitness.calculate_fitness(solution=solution, config_seru=self.config_seru)

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
                    self.config_ga)

                # 进行变异操作
                child_schedule1.schedule_code = GaOperator.swap_mutation(child_schedule1.schedule_code,
                                                                         self.config_ga)
                child_schedule2.schedule_code = GaOperator.swap_mutation(child_schedule2.schedule_code,
                                                                         self.config_ga)

                Initialization.produce_seru_schedule(self.config_seru.num_of_batches, child_schedule1)
                Initialization.produce_seru_schedule(self.config_seru.num_of_batches, child_schedule2)

                # 将变异后的子代添加到 offspring 列表
                offspring.append(child_schedule1)
                offspring.append(child_schedule2)

            best_scheduling: SeruSchedule = min(PSS + offspring, key=lambda x: x.fitness)  # SeruSchedule 类型
            # 更新种群和最佳调度
            PSS = offspring  # List[SeruSchedule]

            # 局部搜索：优化最佳调度
            best_scheduling = GaOperator.local_search_schedule(schedule=best_scheduling, best_formation=formation, config_seru=self.config_seru, config_ccea=self.config_ga)

            current_best_solution = Solution(formation=formation, schedule=best_scheduling)
            CalculateFitness.calculate_fitness(solution=current_best_solution, config_seru=self.config_seru)

            # ------------------------------
            # 3.3 更新当前解
            # ------------------------------
            if current_best_solution.fitness < best_solution.fitness:
                best_solution: Solution = current_best_solution  # Solution 类型

            # 每 100 次迭代输出一次最优解的 fitness 和代数
            if iteration % 5 == 0:
                elapsed_time = time.time() - start_time  # 计算已用时间
                print(f"迭代次数: {iteration}, 适应值: {best_solution.fitness}，耗时: {elapsed_time:.2f}秒")

        # ------------------------------
        # 4. 返回结果
        # ------------------------------
        return best_solution.formation, best_solution.schedule, best_solution

from heuristics.schedule_heuristic import ScheduleHeuristic
def main():
    # 预加载配置
    ConfigLoader.preload_all()

    # 初始化协同进化算法
    ga = GA()

    # 运行协同进化算法
    best_formation, best_scheduling, best_solution = ga.run()

    # FormationHeuristic.formation_heuristic(best_solution)
    ScheduleHeuristic.schedule_heuristic(best_solution)

    # 输出结果
    # print("Best Formation:", best_formation)
    # for i in range(len(best_formation.seru_set)):
    #     print("Seru %d: %s" % (i + 1, len(best_formation.seru_set[i].workers_set)))
    # print("Best Scheduling:", best_scheduling)


# 确保只有在直接运行此文件时才会启动
if __name__ == "__main__":
    main()
