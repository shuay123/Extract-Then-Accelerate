from metaheuristics.ccea.ccea import CCEA
from problem.pure_seru.calculate_fitness_edd import CalculateFitnessEDD
from utils.config_loader import ConfigLoader


def main():
    # 预加载配置
    ConfigLoader.preload_all()

    # 初始化协同进化算法
    ccea = CCEA()

    # 运行协同进化算法
    best_formation, best_scheduling, best_solution = ccea.run()

    print("ccea适应值")
    print(best_solution.fitness)
    CalculateFitnessEDD.calculate_fitness_edd(solution=best_solution, config_seru=ConfigLoader.get_config('config_seru'))
    print("ccea的构造解使用edd的适应值")
    print(best_solution.fitness)


# 确保只有在直接运行此文件时才会启动
if __name__ == "__main__":
    main()
