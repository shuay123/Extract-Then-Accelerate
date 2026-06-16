from copy import deepcopy
import random
from typing import List
from deap import tools

from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.calculate_fitness import CalculateFitness
from problem.pure_seru.calculate_fitness_edd import CalculateFitnessEDD
from problem.pure_seru.initialization import Initialization


class GaOperator:
    @staticmethod
    def tournament_selection(population):

        """
        锦标赛选择：每次从种群中随机选择两个个体，返回适应度更小的个体
        :param population: 当前种群
        :return: 适应度更小的个体
        """
        # 随机选择两个不同的个体
        candidate1, candidate2 = random.sample(population, 2)

        # 返回 makespan 更小的个体
        return candidate1 if candidate1.makespan < candidate2.makespan else candidate2

    @staticmethod
    def order_crossover(parent1: List[int], parent2: List[int], config) -> (List[int], List[int]):
        """
        顺序交叉（调用 DEAP 的 tools.cxOrdered）
        :param parent1: 父代个体 1
        :param parent2: 父代个体 2
        :param config: 配置对象，包含 crossover_rate、mode 和 num_of_workers 字段
        :return: 两个合法的子代个体
        """
        if random.random() < config.crossover_rate:
            # 将两个父代个体的元素减去1（为了调用DEAP）
            parent1_adjusted, parent2_adjusted = map(lambda p: list(map(lambda x: x - 1, p)), [parent1, parent2])

            # 将父代复制为列表，避免直接修改父代
            child1, child2 = parent1_adjusted[:], parent2_adjusted[:]

            # 调用 DEAP 的顺序交叉函数
            tools.cxOrdered(child1, child2)

            # 使用一行代码将两个子代个体的元素加回1
            child1, child2 = map(lambda c: list(map(lambda x: x + 1, c)), [child1, child2])

            # 返回子代
            return child1, child2
        else:
            # 如果没有发生交叉，则直接返回父代
            return parent1, parent2

    @staticmethod
    def swap_mutation(individual: List[int], config) -> List[int]:
        """
        交换变异
        :param individual: 个体
        :param config: 配置对象，包含 mutation_rate、mode 和 num_of_workers 字段
        :return: 变异后的合法个体
        """
        if random.random() < config.mutation_rate:  # 从 config 中获取 mutation_rate
            while True:  # 循环直到生成合法的变异个体
                # 复制个体，避免直接修改原始个体
                mutated_individual = individual.copy()

                # 随机选择两个位置进行交换
                idx1, idx2 = random.sample(range(len(mutated_individual)), 2)

                # 交叉
                mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]

                # 返回变异后的个体
                return mutated_individual
        else:
            # 如果没有发生变异，则直接返回原始个体
            return individual

    @staticmethod
    def local_search_schedule(schedule: "SeruSchedule", best_formation: "SeruFormation", 
                              config_seru, config_ccea, evaluator=None) -> "SeruSchedule":
        """
        局部搜索：优化 SeruSchedule
        :param evaluator: [新增] 自定义评估函数，输入为 solution，返回用于比较的 float 适应度（越小越好）。
                          如果不传，默认使用 CalculateFitness 计算 makespan。
        """
        # 1. 定义评估逻辑：如果有自定义 evaluator 就用自定义的，否则用默认的 makespan
        if evaluator is None:
            def default_evaluator(sol):
                CalculateFitness.calculate_fitness(solution=sol, config_seru=config_seru)
                return sol.makespan
            eval_func = default_evaluator
        else:
            eval_func = evaluator

        best_schedule = schedule
        best_solution = Solution(formation=best_formation, schedule=best_schedule)
        
        # 2. 使用通用评估函数获取 fitness
        best_fitness = eval_func(best_solution)

        for _ in range(config_ccea.m_t):
            neighbor = schedule.__copy__()
            idx1, idx2 = random.sample(range(len(neighbor.schedule_code)), 2)
            neighbor.schedule_code[idx1], neighbor.schedule_code[idx2] = neighbor.schedule_code[idx2], neighbor.schedule_code[idx1]
            Initialization.produce_seru_schedule(config_seru.num_of_batches, neighbor)

            neighbor_solution = Solution(formation=best_formation, schedule=neighbor)
            
            # 3. 计算邻域适应度
            neighbor_fitness = eval_func(neighbor_solution)

            # 4. 比较（通用）
            if neighbor_fitness < best_fitness:
                best_schedule = neighbor
                best_fitness = neighbor_fitness

        return best_schedule

    @staticmethod
    def local_search_formation(formation: "SeruFormation", best_scheduling: "SeruSchedule", 
                               config_seru, config_ccea, evaluator=None) -> "SeruFormation":
        """
        局部搜索：优化 SeruFormation
        """
        # 1. 定义评估逻辑
        if evaluator is None:
            def default_evaluator(sol):
                CalculateFitness.calculate_fitness(solution=sol, config_seru=config_seru)
                return sol.makespan
            eval_func = default_evaluator
        else:
            eval_func = evaluator

        best_formation = formation
        best_solution = Solution(formation=best_formation, schedule=best_scheduling)
        
        # 2. 获取初始 fitness
        best_fitness = eval_func(best_solution)

        for _ in range(config_ccea.m_t):
            neighbor = formation.__copy__()
            idx1, idx2 = random.sample(range(len(neighbor.formation_code)), 2)
            neighbor.formation_code[idx1], neighbor.formation_code[idx2] = neighbor.formation_code[idx2], \
                neighbor.formation_code[idx1]
            Initialization.produce_seru_formation(config_seru.num_of_workers, neighbor)

            neighbor_solution = Solution(formation=neighbor, schedule=best_scheduling)
            
            # 3. 计算邻域适应度
            neighbor_fitness = eval_func(neighbor_solution)

            if neighbor_fitness < best_fitness:
                best_formation = neighbor
                best_fitness = neighbor_fitness

        return best_formation

    @staticmethod
    def local_search_formation_edd(formation: "SeruFormation", best_scheduling: "SeruSchedule", config_seru, config_ccea) -> "SeruFormation":
        """
        局部搜索：优化 SeruFormation 的 formation_code
        :param config_ccea:
        :param config_seru:
        :param formation: 当前 SeruFormation
        :param best_scheduling: 当前最佳 SeruSchedule
        :return: 优化后的 SeruFormation
        """
        best_formation = formation
        best_solution = Solution(formation=best_formation, schedule=best_scheduling)
        CalculateFitnessEDD.calculate_fitness(solution=best_solution, config_seru=config_seru)
        best_fitness = best_solution.makespan  # 使用 makespan 作为适应度值

        for _ in range(config_ccea.m_t):  # 从 config 中获取 m_t

            # 生成邻域解：随机交换 formation_code 中的两个元素
            neighbor = formation.__copy__()
            idx1, idx2 = random.sample(range(len(neighbor.formation_code)), 2)
            neighbor.formation_code[idx1], neighbor.formation_code[idx2] = neighbor.formation_code[idx2], \
                neighbor.formation_code[idx1]
            Initialization.produce_seru_formation(config_seru.num_of_workers, neighbor)

            # 计算邻域解的适应度
            neighbor_solution = Solution(formation=neighbor, schedule=best_scheduling)
            CalculateFitnessEDD.calculate_fitness(solution=neighbor_solution, config_seru=config_seru)
            neighbor_fitness = neighbor_solution.makespan

            # 如果邻域解更优，则更新最佳解
            if neighbor_fitness < best_fitness:
                best_formation = neighbor
                best_fitness = neighbor_fitness

        return best_formation
