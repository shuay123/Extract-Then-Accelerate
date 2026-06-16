from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Seru
import random


class Initialization:

    @staticmethod
    def initial_formation_code(num_of_workers: int) -> list:
        """
        初始化seru构造编码

        生成一个包含工人数的编码列表，并随机打乱，
        用于初始化seru的编码系统。

        :param num_of_workers: 工人的数量
        :return: 返回生成的编码列表
        """
        # formation_code = list(range(1, 2 * num_of_workers ))
        formation_code = list(range(1, 2 * num_of_workers + 1))
        random.shuffle(formation_code)
        return formation_code  # 返回生成的编码列表

    @staticmethod
    def initial_schedule_code(num_of_workers: int, num_of_batches: int) -> list:
        """
        初始化seru调度编码

        生成一个包含工人和批次数量的编码列表，并随机打乱，
        用于初始化调度系统。

        :param num_of_workers: 工人的数量
        :param num_of_batches: 批次的数量
        :return: 返回生成的调度编码列表
        """
        schedule_code = list(range(1, num_of_workers + num_of_batches))
        random.shuffle(schedule_code)
        return schedule_code  # 返回生成的编码列表

    @staticmethod
    def produce_seru_formation(num_of_workers: int, formation: 'SeruFormation') -> None:
        """
        初始化seru构造

        根据formation_code生成seru构造集，每个seru包含一组工人。
        如果一个编码属于工人，则添加工人到当前seru，其他则创建新的seru。

        :param num_of_workers: 工人的数量
        :param formation: SeruFormation对象，用于存储生成的seru构造
        :return: None
        """
        formation.seru_set = []  # 存储生成的所有seru
        current_seru = Seru()  # 创建新的seru实例
        # 遍历formation_code，并根据编码分配工人
        for code in formation.formation_code:
            if code <= num_of_workers:
                current_seru.workers_set.append(code)  # 工人编码从1开始
            else:
                if current_seru.workers_set:
                    formation.seru_set.append(current_seru)  # 如果当前seru有工人，添加到seru_set
                current_seru = Seru()  # 创建新seru

        if current_seru.workers_set:
            formation.seru_set.append(current_seru)  # 确保最后一个seru也被添加

    @staticmethod
    def produce_seru_schedule(num_of_batches: int, schedule: 'SeruSchedule') -> None:
        """
        初始化seru调度

        根据调度编码生成batches_assignment列表，指示每个批次的调度情况。

        :param num_of_batches: 批次的数量
        :param schedule: SeruSchedule对象，用于存储生成的批次调度
        :return: None
        """
        schedule.batches_assignment = []  # 存储所有批次调度
        batches = []  # 用于临时存储当前批次

        # 遍历调度编码，并分配批次
        for batch in schedule.schedule_code:
            if batch <= num_of_batches:
                batches.append(batch)  # 批次从1开始
            else:
                if batches:
                    schedule.batches_assignment.append(batches)  # 将当前批次添加到batches_assignment
                batches = []  # 创建新的批次

        if batches:
            schedule.batches_assignment.append(batches)  # 确保最后一个批次也被添加
