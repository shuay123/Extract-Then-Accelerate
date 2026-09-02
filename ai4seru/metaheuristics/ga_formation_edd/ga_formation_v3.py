import sys
import os
# 添加项目根目录到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import time

import time
from typing import List
from metaheuristics.common.ga_operator import GaOperator
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.initialization import Initialization
# from problem.pure_seru.calculate_fitness import CalculateFitness
from problem.pure_seru.calculate_fitness_edd import CalculateFitnessEDD
from utils.config_loader import ConfigLoader
from utils.excel_utils import ExcelDataLoader


class GA:
    def __init__(self, worker_map, batch_map):
        """
        初始化协同进化算法
        """
        # 加载配置
        self.config_seru = ConfigLoader.get_config('config_seru')
        loader = ExcelDataLoader.instance()
        self.config_ga = ConfigLoader.get_config('config_ga')
        loader = ExcelDataLoader.instance()
        self.archive: List[SeruFormation] = []

        self.config_seru.worker_map = worker_map

        self.config_seru.selected_batches = list(batch_map.values())
        if hasattr(self.config_seru, 'batch_map'):
            delattr(self.config_seru, 'batch_map')

        # 初始化 Seru 数据
        self.loader = ExcelDataLoader()
        self.loader.read_data(excel_path=self.config_seru.seru_data_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.batch_types_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.due_dates_path, config_sheet=self.config_seru)

    def _update_archive(self, candidates: List[SeruFormation]) -> None:
        """
        仅按 fitness 去重：
        1. 合并当前 archive 与候选个体
        2. 按 fitness 从小到大排序（越小越好）
        3. 如果多个个体 fitness 相同，只保留遇到的第一个
        4. 截取前 archive_size 个加入 self.archive
        """
        # ① 合并并排序
        merged = self.archive + [c.__copy__() for c in candidates]
        merged.sort(key=lambda x: x.fitness)  # 升序：最优排前面

        # ② 去重（只看 fitness）
        uniq: List[SeruFormation] = []
        seen_fitness = set()
        for fm in merged:
            if fm.fitness not in seen_fitness:
                uniq.append(fm)
                seen_fitness.add(fm.fitness)
            if len(uniq) == self.config_ga.archive_size:  # 足够就停
                break

        # ③ 更新 archive
        self.archive = uniq

    def run(self, ) -> (SeruFormation, SeruSchedule, Solution,List[Solution]):
        """
        运行协同进化算法
        :return: 最佳 Seru 构造、最佳 Seru 调度、最佳解
        """
        # ------------------------------
        # 1. 初始化种群
        # ------------------------------
        # 验证 formation_code 有效性
        num_workers = self.config_seru.num_of_workers

        # 初始化 Seru 构造
        schedule = SeruSchedule()

        # 初始化 Seru 调度种群
        PSF: List[SeruFormation] = []  # 显式指定 PSF 是一个 SeruFormation 类型的列表
        for _ in range(self.config_ga.population_size):
            formation_code = Initialization.initial_formation_code(self.config_seru.num_of_workers)
            formation = SeruFormation(formation_code=formation_code)
            Initialization.produce_seru_formation(self.config_seru.num_of_workers, formation)
            PSF.append(formation)


        best_solution: Solution = Solution(formation=PSF[0], schedule=schedule)
        CalculateFitnessEDD.calculate_fitness(solution=best_solution, config_seru=self.config_seru)
        # CalculateFitness.calculate_fitness(solution=best_solution, config_seru=self.config_seru)


        # ------------------------------
        # 3. 主循环
        # ------------------------------
        start_time = time.time()  # 记录算法开始时间
        iteration = 0  # 迭代计数器
        while iteration < self.config_ga.max_iteration:
            iteration += 1  # 更新迭代计数器
            # ------------------------------
            # 3.2 进化 Seru 调度种群
            # ------------------------------
            # 计算适应度
            for formation in PSF:  # PSF: List[SeruFormation]
                current_schedule = SeruSchedule()
                solution: Solution = Solution(formation=formation, schedule=current_schedule)  # Solution 类型
                CalculateFitnessEDD.calculate_fitness(solution, self.config_seru)
                # CalculateFitness.calculate_fitness(solution, self.config_seru)

            self._update_archive(PSF)

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
                    self.config_ga)

                # 进行变异操作
                child_formation1.formation_code = GaOperator.swap_mutation(child_formation1.formation_code,
                                                                           self.config_ga)
                child_formation2.formation_code = GaOperator.swap_mutation(child_formation2.formation_code,
                                                                           self.config_ga)
                
                Initialization.produce_seru_formation(self.config_seru.num_of_workers, child_formation1)
                Initialization.produce_seru_formation(self.config_seru.num_of_workers, child_formation2)

                # 将变异后的子代添加到 offspring 列表
                offspring.append(child_formation1)
                offspring.append(child_formation2)

            for child_formation in offspring:
                # 创建临时 Solution 对象用于计算
                temp_schedule = SeruSchedule()
                temp_solution = Solution(formation=child_formation, schedule=temp_schedule)
                
                # 计算适应度 (更新 child_formation.fitness 和 makespan)
                CalculateFitnessEDD.calculate_fitness(solution=temp_solution, config_seru=self.config_seru)
            
            
            # 1. 精英保留策略 (Elitism)
            # 在父代消失前，必须把父代中最好的那个保存下来
            parent_elite = min(PSF, key=lambda x: x.fitness).__copy__()

            # 2. 更新全局最佳 (用于局部搜索和最终输出)
            # 找出当前这一代(含子代)中最优秀的，作为局部搜索的种子
            current_all = PSF + offspring
            best_formation = min(current_all, key=lambda x: x.fitness).__copy__()

            # 3. 代际替换 (Generational Replacement)
            # Replace parents directly with offspring instead of sorting a mixed pool.
            # 这强迫算法跳出局部最优，去探索子代的新空间
            PSF = offspring 

            # 4. 精英回插
            # 将保存的父代精英，替换掉新一代(子代)中最差的一个
            # 保证种群质量下限不会比上一代更差
            worst_child_idx = PSF.index(max(PSF, key=lambda x: x.fitness))
            PSF[worst_child_idx] = parent_elite

            # 5. Diversity monitoring and forced mutation
            # 检查种群中 fitness 不同的个体数量
            unique_fitness_values = set(ind.fitness for ind in PSF)
            
            # 如果种群中 90% 的个体 fitness 都一样（陷入早熟收敛）
            if len(unique_fitness_values) < max(2, self.config_ga.population_size * 0.1):
                # print(f"    >> 检测到种群同质化 (Unique Fitness: {len(unique_fitness_values)})，执行灾变变异...")
                # 保留前 10% 的精英，对后 90% 进行强制大变异
                num_elites = int(self.config_ga.population_size * 0.1)
                # 先排序
                PSF.sort(key=lambda x: x.fitness)
                
                for k in range(num_elites, len(PSF)):
                    # 进行多次交换变异，破坏现有结构
                    for _ in range(2): 
                        PSF[k].formation_code = GaOperator.swap_mutation(PSF[k].formation_code, self.config_ga)
                    
                    # 变异后必须重新计算结构和 fitness
                    Initialization.produce_seru_formation(self.config_seru.num_of_workers, PSF[k])
                    temp_sol_mut = Solution(formation=PSF[k], schedule=SeruSchedule())
                    CalculateFitnessEDD.calculate_fitness(temp_sol_mut, self.config_seru)
            
            # ---------------------------------------------------------------------

            # 打印排序后的 PSF 列表，验证排序结果
            # PSF[0]=best_formation.__copy__()

            # 局部搜索：优化最佳构造
            temp_schedule_for_search = SeruSchedule()
            best_formation = GaOperator.local_search_formation_edd(formation=best_formation, best_scheduling=temp_schedule_for_search, config_ccea=self.config_ga, config_seru=self.config_seru)
            new_schedule_for_update = SeruSchedule()
            current_best_solution = Solution(formation=best_formation.__copy__(), schedule=new_schedule_for_update)
            CalculateFitnessEDD.calculate_fitness(solution=current_best_solution, config_seru=self.config_seru)
            # CalculateFitness.calculate_fitness(solution=current_best_solution, config_seru=self.config_seru)

            PSF[0] = current_best_solution.formation.__copy__()
            # 使用 enumerate 获取序号（从1开始），并用列表推导式生成格式化的字符串

            # ------------------------------
            # 3.3 更新当前解
            # ------------------------------
            if current_best_solution.fitness < best_solution.fitness:
                best_solution: Solution = current_best_solution.__copy__()  # Solution 类型



            # 每 100 次迭代输出一次最优解的 fitness 和代数
            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time  # 计算已用时间
                print(f"迭代次数: {iteration}, 适应值: {best_solution.fitness}，耗时: {elapsed_time:.2f}秒")

        # 4. 返回结果
        # ------------------------------
        return best_solution.formation, best_solution.schedule, best_solution,self.archive, self.config_seru
import random

def main():
    # random.seed(42)
    # 预加载配置
    ConfigLoader.preload_all()
    config_seru = ConfigLoader.get_config('config_seru')
    all_real_workers = list(range(1, 41))   
    all_real_batches = list(range(1, 501))

    # 随机采样
    selected_workers = random.sample(all_real_workers, config_seru.num_of_workers)
    selected_batches = random.sample(all_real_batches, config_seru.num_of_batches)

    # 建立映射字典：逻辑ID(1~N) -> 真实ID(Random)
    # Logical IDs start at 1 to match Initialization.
    worker_map = {
        logic_id + 1: real_id 
        for logic_id, real_id in enumerate(selected_workers)
    }
    batch_map = {
        logic_id + 1: real_id
        for logic_id, real_id in enumerate(selected_batches)
    }
    print(f"随机映射工人: {worker_map}")
    print(f"随机选取批次: {batch_map}")
    # 初始化GA
    ga = GA(worker_map, batch_map)

    # 运行GA
    best_formation, best_scheduling, best_solution,archive, config_seru = ga.run()

    # FormationHeuristic.formation_heuristic(best_solution)
    print(best_solution)



# 确保只有在直接运行此文件时才会启动
if __name__ == "__main__":
    main()
