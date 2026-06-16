from problem.pure_seru.pure_seru_entities import Solution, Seru
from utils.excel_utils import ExcelDataLoader
from problem.pure_seru.calculate_fitness import CalculateFitness
from utils.config_loader import ConfigLoader


class ScheduleHeuristic:
    @staticmethod
    def schedule_heuristic(original_solution: Solution) -> Solution:
        config_seru = ConfigLoader.get_config('config_seru')
        excel_loader = ExcelDataLoader()
        CalculateFitness.calculate_fitness(original_solution, config_seru)
        """Seru调度启发式算法，优化批次处理顺序"""
        # 深拷贝解并清空批次分配记录
        new_solution = original_solution.__copy__()
        new_solution.schedule.batches_assignment.clear()
        # 遍历所有Seru单元进行批次排序
        for seru in new_solution.formation.seru_set:
            ScheduleHeuristic._reschedule_seru_batches(seru, excel_loader)
        # 重新计算适应值
        CalculateFitness.calculate_fitness(new_solution, config_seru)
        print(f"Original Tardiness: {original_solution.fitness:.2f} => Optimized Tardiness: {new_solution.fitness:.2f}")
        return new_solution

    @staticmethod
    def _reschedule_seru_batches(seru: Seru, excel_loader: ExcelDataLoader):
        """对单个Seru单元进行批次重排序（修正版）"""
        # 使用有序字典保持产品类型顺序
        from collections import defaultdict
        type_dict = defaultdict(list)

        # 收集批次信息
        for idx, batch_id in enumerate(seru.batches_set):
            product_type = excel_loader.batch_to_product_dict[batch_id]['产品类型']
            due_date = excel_loader.batch_due_dates_dict[batch_id]['批次截止时间']
            type_dict[product_type].append((due_date, batch_id, idx))

        # 创建副本用于安全修改
        new_batches = seru.batches_set.copy()

        # 对每个产品类型进行排序
        for product_type, batches in type_dict.items():
            if len(batches) < 2:
                continue  # 单批次无需排序

            # 按截止时间升序排序 (最早due的排前面)
            sorted_batches = sorted(batches, key=lambda x: x[0])

            # 提取排序后的批次ID序列
            sorted_ids = [b[1] for b in sorted_batches]

            # 获取该类型所有批次原始位置(保持不同类型批次的相对位置)
            original_positions = [b[2] for b in batches]  # 原无序位置

            # 将排序后的ID填入原始位置
            for pos, bid in zip(original_positions, sorted_ids):
                new_batches[pos] = bid

        # 更新Seru的批次序列
        seru.batches_set[:] = new_batches
