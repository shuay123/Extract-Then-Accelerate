from problem.pure_seru.pure_seru_entities import Solution, Seru
from utils.excel_utils import ExcelDataLoader
from utils.config_loader import ConfigLoader
from problem.pure_seru.calculate_fitness import CalculateFitness


class FormationHeuristic:

    @staticmethod
    def formation_heuristic(original_solution: Solution) -> Solution:
        config_seru = ConfigLoader.get_config('config_seru')
        # 深拷贝原始解
        new_solution = original_solution.__copy__()
        excel_loader = ExcelDataLoader.instance()


        # 获取所有Seru单元并保存原始工人数量
        seru_list = new_solution.formation.seru_set
        original_worker_counts = [len(seru.workers_set) for seru in seru_list]

        # 清空所有Seru的工人并收集所有工人
        all_workers = []
        for seru in seru_list:
            all_workers.extend(seru.workers_set)
            seru.workers_set.clear()

        # 按原始工人数量排序Seru（从小到大）
        sorted_serus = sorted(
            zip(seru_list, original_worker_counts),
            key=lambda x: x[1]
        )

        # 重新分配工人
        for seru, worker_count in sorted_serus:
            # 获取该Seru所有批次的产品类型
            product_types = []
            for batch_id in seru.batches_set:
                batch_info = excel_loader.batch_to_product_dict.get(batch_id)
                if not batch_info:
                    raise ValueError(f"Batch {batch_id} not found")
                product_types.append(batch_info['产品类型'])

            # 没有批次的Seru随机分配工人
            if not product_types:
                seru.workers_set.extend(all_workers[:worker_count])
                del all_workers[:worker_count]
                continue

            # 为每个工人槽位选择最佳工人
            for _ in range(worker_count):
                best_worker = None
                # 初始化一个"无限大"的基准值
                min_avg_time = float('inf')

                # 遍历剩余工人计算适配度
                for worker in all_workers:
                    total_time = 0
                    for pt in product_types:
                        worker_time = excel_loader.worker_to_product_dict.get(worker, {}).get(pt, float('inf'))
                        total_time += worker_time
                    avg_time = total_time / len(product_types)

                    if avg_time < min_avg_time:
                        min_avg_time = avg_time
                        best_worker = worker

                # 分配最佳工人并更新工人池
                if best_worker:
                    seru.workers_set.append(best_worker)
                    all_workers.remove(best_worker)

        # 重新计算适应度
        CalculateFitness.calculate_fitness(new_solution, config_seru)
        # Assign fitness values to the original and candidate solutions.
        print(f"Original Solution Fitness: {original_solution.fitness}")
        print(f"New Solution Fitness: {new_solution.fitness}")
        return new_solution



