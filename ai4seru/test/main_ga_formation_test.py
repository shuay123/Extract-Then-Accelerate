from metaheuristics.ga_formation_edd.ga_formation import GA
from utils.config_loader import ConfigLoader

"""
构造已知，Seru调度使用GA
"""


def main():
    # 预加载配置
    ConfigLoader.preload_all()

    # 初始化协同进化算法

    for i in range(20):
        ga = GA()
        # 运行协同进化算法
        best_formation, best_scheduling, best_solution = ga.run()
        print(best_solution.fitness)

    # 输出结果
    # print("Best Formation:", best_formation)
    # for i in range(len(best_formation.seru_set)):
    #     print("Seru %d: %s" % (i + 1, len(best_formation.seru_set[i].workers_set)))
    # print("Best Scheduling:", best_scheduling)


# 确保只有在直接运行此文件时才会启动
if __name__ == "__main__":
    main()
