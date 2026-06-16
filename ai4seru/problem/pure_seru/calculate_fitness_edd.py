from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Seru, Solution
from problem.pure_seru.calculate_fitness import CalculateFitness
from utils.excel_utils import ExcelDataLoader


class CalculateFitnessEDD:
    @staticmethod
    def calculate_fitness(solution: Solution, config_seru):
        """
        计算解的适应度
        :param config_seru:
        :param solution: 解对象
        """
        formation = solution.formation
        schedule = solution.schedule
        excel_loader = ExcelDataLoader.instance()
        # 初始化
        for seru in formation.seru_set:
            seru.batches_set.clear()
        CalculateFitness.init(formation, schedule)
        # 分配批次并计算流动时间和劳动时间
        CalculateFitnessEDD.calculate_total_throughput_time(schedule=schedule, formation=formation, config_seru=config_seru, excel_loader=excel_loader)
        # 更新解的各项目标值
        solution.makespan = schedule.makespan = formation.makespan = round(max(seru.throughput_time for seru in formation.seru_set), 3)
        solution.labour_time = schedule.labour_time = formation.labour_time = round(sum(seru.labour_time for seru in formation.seru_set), 3)
        solution.tardiness = schedule.tardiness = formation.tardiness = round(sum(seru.tardiness for seru in formation.seru_set), 3)
        # 设定目标值
        solution.fitness = schedule.fitness = formation.fitness = solution.makespan

    @staticmethod
    def calculate_total_throughput_time(schedule: SeruSchedule, formation: SeruFormation, config_seru, excel_loader):
        """
        计算批次在 Seru 单元中的流动时间和劳动时间
        :param config_seru:
        :param formation:
        :param schedule: Seru 调度
        """
        target_batches = getattr(config_seru, 'selected_batches', range(1, config_seru.num_of_batches + 1))
        
        for batch_id in target_batches:
            # 找到时间最短的Seru
            seru_index = min(range(len(formation.seru_set)), key=lambda idx: formation.seru_set[idx].throughput_time)
            seru = formation.seru_set[seru_index]
            # 将批次添加到 Seru 单元
            seru.batches_set.append(batch_id)
            # 计算单个batch的throughput
            throughput_time_of_batch = CalculateFitness.calculate_batch_throughput_time(batch_id=batch_id, config_seru= config_seru, excel_loader= excel_loader, seru= seru)

            # 换装时间
            setup_time = CalculateFitness.calculate_setup_time(seru=seru, config_seru=config_seru, excel_loader=excel_loader)

            # 更新 Seru 单元的总流动时间，不考虑换装时间
            # seru.throughput_time += throughput_time_of_batch

            # 更新 Seru 单元的总流动时间，考虑换装时间
            # seru.throughput_time += (throughput_time_of_batch + setup_time)
            seru.throughput_time += (throughput_time_of_batch)

            # seru.tardiness += max(0, seru.throughput_time - excel_loader.batch_due_dates_dict.get(batch_id)["批次截止时间"])
            seru.labour_time += throughput_time_of_batch * len(seru.workers_set)
            # 记录批次的流动时间
            schedule.batches_throughput_time_in_seru.append((batch_id, seru.throughput_time))
