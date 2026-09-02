from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Seru, Solution
from utils.excel_utils import ExcelDataLoader


class CalculateFitness:
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
        # 初始化 Seru 单元和批次流动时间记录
        if schedule.batches_assignment:
            for seru in formation.seru_set:
                seru.batches_set.clear()
        CalculateFitness.init(formation, schedule)
        # 分配批次并计算流动时间和劳动时间
        CalculateFitness.calculate_total_throughput_time(schedule=schedule, formation=formation, config_seru=config_seru, excel_loader=excel_loader)
        # 更新解的各项目标值
        solution.makespan = schedule.makespan = formation.makespan = round(max(seru.processing_time for seru in formation.seru_set), 3)
        solution.labour_time = schedule.labour_time = formation.labour_time = round(sum(seru.labour_time for seru in formation.seru_set), 3)
        solution.tardiness = schedule.tardiness = formation.tardiness = round(sum(seru.tardiness for seru in formation.seru_set), 3)
        # 设定目标值
        solution.fitness = schedule.fitness = formation.fitness = solution.makespan

    @staticmethod
    def init(formation, schedule):
        for seru in formation.seru_set:
            seru.throughput_time = 0.0
            seru.labour_time = 0.0
            seru.tardiness = 0.0
            seru.processing_time = 0.0
        schedule.batches_throughput_time_in_seru.clear()


    @staticmethod
    def calculate_total_throughput_time(schedule: SeruSchedule, formation: SeruFormation, config_seru, excel_loader):
        """
        计算所有batches的流通时间
        :param excel_loader:
        :param config_seru:
        :param formation:
        :param schedule: Seru 调度
        """
        if schedule.batches_assignment:
            for i in range(len(schedule.batches_assignment)):
                batches = schedule.batches_assignment[i]
                seru = formation.seru_set[i % len(formation.seru_set)]
                seru.batches_set.extend(batches)
                CalculateFitness.calculate_throughput_time_in_seru(batches, config_seru, excel_loader, schedule, seru)
        else:
            for seru in formation.seru_set:
                CalculateFitness.calculate_throughput_time_in_seru(seru.batches_set, config_seru, excel_loader, schedule, seru)
    
    @staticmethod
    def calculate_throughput_time_in_seru(batches, config_seru, excel_loader, schedule, seru):
        
        
        for batch_id in batches:
            # 将批次添加到 Seru 单元
            real_batch_id = batch_id
            if hasattr(config_seru, 'batch_map'):
                real_batch_id = config_seru.batch_map.get(batch_id, batch_id)
            # 计算单个batch的throughput
            throughput_time_of_batch = CalculateFitness.calculate_batch_throughput_time(seru=seru, batch_id=batch_id, config_seru=config_seru, excel_loader=excel_loader)

            # 换装时间
            setup_time = CalculateFitness.calculate_setup_time(seru=seru, config_seru=config_seru, excel_loader=excel_loader)

            # 打印每个批次的 throughput_time_of_batch 和 setup_time
            # print(f"Batch ID: {batch_id}, Throughput Time: {throughput_time_of_batch}, Setup Time: {setup_time}")

            # 更新 Seru 单元的总流动时间，不考虑换装时间
            # seru.throughput_time += throughput_time_of_batch

            # 更新 Seru 单元的总流动时间，考虑换装时间
            ttb = 0.0 if throughput_time_of_batch is None else float(throughput_time_of_batch)
            st = 0.0 if setup_time is None else float(setup_time)
            seru.throughput_time += (ttb + st)
            seru.processing_time += throughput_time_of_batch

            due_info = excel_loader.batch_due_dates_dict.get(real_batch_id) if excel_loader.batch_due_dates_dict else None
            if due_info and due_info.get("批次截止时间") is not None:
                seru.tardiness += max(0, seru.throughput_time - float(due_info["批次截止时间"]))
            # 打印batch_id seru.throughput_time excel_loader.batch_due_dates_dict.get(batch_id)["批次截止时间"] 和 tardiness
            # print(f"batch_id: {batch_id},  seru.throughput_time: {seru.throughput_time}, batch_due_date: {excel_loader.batch_due_dates_dict.get(batch_id)['批次截止时间']}, tardiness: {seru.tardiness}")
            seru.labour_time += throughput_time_of_batch * len(seru.workers_set)
            # 记录批次的流动时间
            schedule.batches_throughput_time_in_seru.append((batch_id, seru.throughput_time))

    @staticmethod
    def calculate_batch_throughput_time(seru: Seru, batch_id: int, config_seru, excel_loader):
         # 判断使用哪种计算逻辑
        use_standard_logic = hasattr(config_seru, 'use_standard_logic') and config_seru.use_standard_logic
        real_batch_id = batch_id
        if hasattr(config_seru, 'batch_map'):
            real_batch_id = config_seru.batch_map.get(batch_id, batch_id)
        # use_standard_logic为true打印使用标准逻辑，反之打印使用企业逻辑
        # if use_standard_logic:
        #     print("使用标准逻辑")
        # else:
        #     print("使用企业逻辑")
        if real_batch_id in excel_loader.batch_to_product_dict:
            product_type_id = excel_loader.batch_to_product_dict[real_batch_id]['产品类型']
            batch_size = excel_loader.batch_to_product_dict[real_batch_id]['批次大小']
        else:
            # 抛出异常时显示详细信息
            raise ValueError(f"Batch ID {real_batch_id} (Logic: {batch_id}) not found in batch_to_product_dict")
        
        task_time_in_seru = 0.0

        for worker_id in seru.workers_set:
            real_worker_id = worker_id
            if hasattr(config_seru, 'worker_map'):
                real_worker_id = config_seru.worker_map.get(worker_id, worker_id)
            # 从字典中获取工人的多能工系数
            if real_worker_id in excel_loader.worker_to_task_dict:
                worker_coefficient_of_multiple_task = excel_loader.worker_to_task_dict[real_worker_id]['系数']
            else:
                raise ValueError(f"Worker ID {real_worker_id} not found in worker_to_task_dict")
            
            c = 1.0

            # 计算多能工系数
            if (config_seru.num_of_workers - config_seru.max_num_of_multiple_task) > 0:
                c += worker_coefficient_of_multiple_task * (config_seru.num_of_workers - config_seru.max_num_of_multiple_task)

            # 从字典中获取工人对产品类型的熟练程度
            if real_worker_id in excel_loader.worker_to_product_dict:
                if product_type_id in excel_loader.worker_to_product_dict[real_worker_id]:
                    worker_to_product_type_coefficient = excel_loader.worker_to_product_dict[real_worker_id][product_type_id]
                else:
                    raise ValueError(f"Product type ID {product_type_id} not found for worker ID {real_worker_id}")
            else:
                raise ValueError(f"Worker ID {real_worker_id} not found in worker_to_product_dict")

            # 根据配置决定是否包含 TASK_TIME
            if use_standard_logic:
                # 标准逻辑：包含 TASK_TIME 常量
                task_time = config_seru.task_time
                task_time_in_seru += task_time * c * worker_to_product_type_coefficient
            else:
                # 企业逻辑：不包含 TASK_TIME
                task_time_in_seru += c * worker_to_product_type_coefficient

        # 计算平均任务时间
        task_time_in_seru /= len(seru.workers_set)
        # 计算批次的流动时间
        flow_time_of_batch = task_time_in_seru * batch_size * config_seru.num_of_workers / len(seru.workers_set)
        return flow_time_of_batch

    # @staticmethod
    # def calculate_setup_time(seru: Seru, config_seru, excel_loader):
    #     """
    #     计算换装时间
    #     :param seru: Seru 单元对象
    #     :param config_seru: 配置对象
    #     :param excel_loader: Excel 数据加载器
    #     :return: 换装时间
    #     """
    #
    #     # 判断 seru.batches_set 的元素个数
    #     if len(seru.batches_set) < 2:
    #         return 0
    #     # 获取最后两个元素的 batch_id
    #     last_batch_id = seru.batches_set[-1]
    #     second_last_batch_id = seru.batches_set[-2]
    #     # print(f"last_batch_id: {last_batch_id}, second_last_batch_id: {second_last_batch_id}")
    #
    #     # 获取 batch_id 对应的产品类型
    #     if last_batch_id in excel_loader.batch_to_product_dict and second_last_batch_id in excel_loader.batch_to_product_dict:
    #         last_product_type = excel_loader.batch_to_product_dict[last_batch_id]['产品类型']
    #         second_last_product_type = excel_loader.batch_to_product_dict[second_last_batch_id]['产品类型']
    #         # print(f"last_product_type: {last_product_type}, second_last_product_type: {second_last_product_type}")
    #
    #         # 判断产品类型是否相同
    #         if last_product_type == second_last_product_type:
    #             return 0
    #         else:
    #             return config_seru.setup_time

    @staticmethod
    def calculate_setup_time(seru: Seru, config_seru, excel_loader, batch = None):
        return 0


