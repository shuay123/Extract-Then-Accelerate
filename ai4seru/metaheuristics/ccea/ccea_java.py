# metaheuristics/ccea/ccea.py
# python -m metaheuristics.ccea.ccea_java
import time
from typing import List
from utils.excel_utils import ExcelDataLoader
from metaheuristics.common.ga_operator import GaOperator

# 纯seru生产则使用下面的导入
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.initialization import Initialization
from problem.pure_seru.calculate_fitness import CalculateFitness

from flask import Flask, request, jsonify
import json
from utils.config_loader import ConfigLoader


# 🟢 [新增] 一个辅助类，用于将字典转换为对象
# 这样就不需要修改 CCEA 内部所有 `self.config_seru.xxx` 的调用
class SimpleConfig:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
    
    def __getattr__(self, name):
        # 如果 Java 传入的 JSON 缺少某个键，返回 None 而不是崩溃
        return self.__dict__.get(name)

class CCEA:
    def __init__(self, config_seru_dict: dict, problem_data_dict: dict):
        """
        初始化协同进化算法
        """
        # 加载配置
        self.config_seru = SimpleConfig(config_seru_dict)
        self.config_ccea = ConfigLoader.get_config('config_ccea')

        # 初始化 Seru 数据
        self.loader = ExcelDataLoader.instance()
        self.loader.load_data_from_dicts(
            worker_task_data=problem_data_dict.get("worker_to_task_dict", {}),
            worker_prod_data=problem_data_dict.get("worker_to_product_dict", {}),
            batch_prod_data=problem_data_dict.get("batch_to_product_dict", {}),
            batch_due_dates_data=problem_data_dict.get("batch_due_dates_dict", {})
        )
        # 检查关键数据是否存在
        if not self.config_seru.num_of_workers or not self.config_seru.num_of_batches:
            raise ValueError("config_seru_dict 必须包含 'num_of_workers' 和 'num_of_batches'")

        # 校验批次覆盖范围与键类型一致性
        expected_ids = set(range(1, int(self.config_seru.num_of_batches) + 1))
        actual_ids = set(self.loader.batch_to_product_dict.keys())
        missing = sorted(list(expected_ids - actual_ids))
        if missing:
            raise ValueError(f"batch_to_product_dict 缺少以下批次ID: {missing[:20]}{' 等' if len(missing) > 20 else ''}")

    def run(self) -> (SeruFormation, SeruSchedule, Solution):
        """
        运行协同进化算法
        :return: 最佳 Seru 构造、最佳 Seru 调度、最佳解
        """
        # ------------------------------
        # 1. 初始化种群
        # ------------------------------
        # 初始化 Seru 构造种群
        PSF: List[SeruFormation] = []  # 显式指定 PSF 是一个 SeruFormation 类型的列表
        for _ in range(self.config_ccea.population_size):
            formation_code = Initialization.initial_formation_code(self.config_seru.num_of_workers)
            formation = SeruFormation(formation_code=formation_code)
            Initialization.produce_seru_formation(self.config_seru.num_of_workers, formation)
            PSF.append(formation)

        # 初始化 Seru 调度种群
        PSS: List[SeruSchedule] = []  # 显式指定 PSS 是一个 SeruSchedule 类型的列表
        for _ in range(self.config_ccea.population_size):
            schedule_code = Initialization.initial_schedule_code(self.config_seru.num_of_workers, self.config_seru.num_of_batches)
            schedule = SeruSchedule(schedule_code=schedule_code)
            Initialization.produce_seru_schedule(self.config_seru.num_of_batches, schedule)
            PSS.append(schedule)

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
        while time.time() - start_time < self.config_ccea.max_runtime:
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
            if iteration % 1 == 0:
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


app = Flask(__name__)

# 预加载算法配置 (config_ccea)
try:
    ConfigLoader.preload_all()
    print("CCEA 算法配置已加载。")
except Exception as e:
    print(f"警告：预加载配置失败: {e}")

@app.route('/run_ccea', methods=['POST'])
def run_ccea_endpoint():
    """
    运行 CCEA 算法的 API 端点
    Java 将向此端点发送一个 POST 请求
    """
    print("收到 /run_ccea 请求...")
    try:
        # 1. 从 Java 获取 JSON 数据
        input_data = request.get_json(silent=True)
        if input_data is None:
            return jsonify({
                "error": "请求体必须是合法 JSON，并设置 Content-Type=application/json"
            }), 400
        
        # 2. 分离配置数据和问题数据
        config_seru_data = input_data.get("config_seru")
        problem_data = input_data.get("problem_data")
        
        if not config_seru_data or not problem_data:
            return jsonify({"error": "请求体 JSON 必须包含 'config_seru' 和 'problem_data' 键"}), 400

        # 3. 预检入参一致性
        print("正在初始化 CCEA 实例...")
        try:
            num_batches = int(config_seru_data.get("num_of_batches"))
        except Exception:
            return jsonify({"error": "config_seru.num_of_batches 必须为整数"}), 400

        batch_dict = problem_data.get("batch_to_product_dict", {})
        # 统一将可转换的键转为整数进行范围校验
        key_set = set()
        for k in batch_dict.keys():
            try:
                key_set.add(int(k))
            except Exception:
                # 非数字键忽略参与范围判断，但会在加载阶段抛出更详细错误
                pass
        expected = set(range(1, num_batches + 1))
        missing_ids = sorted(list(expected - key_set))
        if missing_ids:
            return jsonify({
                "error": "batch_to_product_dict 未完整覆盖所需批次ID",
                "num_of_batches": num_batches,
                "missing_batch_ids": missing_ids[:50]
            }), 400
        # 4. 初始化 CCEA (这将自动注入数据到 ExcelDataLoader)
        ccea = CCEA(config_seru_dict=config_seru_data, problem_data_dict=problem_data)
        
        # 5. 运行算法
        print("CCEA 正在运行...")
        best_formation, best_scheduling, best_solution = ccea.run()
        print("CCEA 运行完成。")

        # 6. 将结果格式化为 JSON 返回给 Java
        result_dict = {
            "fitness": best_solution.fitness,
            "makespan": best_solution.makespan,
            "labour_time": best_solution.labour_time,
            "formation_code": best_solution.formation.formation_code,
            "schedule_code": best_solution.schedule.schedule_code,
            "formation_details": [
                {"seru_id": i + 1, "workers": seru.workers_set}
                for i, seru in enumerate(best_solution.formation.seru_set)
            ],
            "schedule_details": [
                {"assignment_group": i + 1, "batches": batches}
                for i, batches in enumerate(best_solution.schedule.batches_assignment)
            ]
        }
        
        return jsonify(result_dict), 200

    except Exception as e:
        print(f"执行 CCEA 时发生错误: {e}")
        return jsonify({"error": str(e)}), 500

# 确保只有在直接运行此文件时才会启动服务器
if __name__ == "__main__":
    print("启动 Flask 服务器，监听 http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False) # 使用 debug=False 提高性能
