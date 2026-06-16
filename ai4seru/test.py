from metaheuristics.ga_formation_edd.ga_formation_v3 import GA
from metaheuristics.ga_formation_edd.ga_formation_v3_2 import GA2
from metaheuristics.ccea.ccea import CCEA
from utils.excel_output import save_serudata_to_excel
from utils.config_loader import ConfigLoader
from problem.pure_seru.pure_seru_entities import Solution
import random
from utils.excel_utils import ExcelDataLoader
from typing import List, Dict, Any



def generate_result_data_structure(solution: Solution, config_seru, run_idx=1) -> List[Dict[str, Any]]:
    """
    根据最优解和配置，生成用于保存到Excel的结构化数据列表。
    适配 ExcelDataLoader 的数据结构。
    """
    excel_loader = ExcelDataLoader.instance()
    
    # ---------------------------------------------------------
    # 0. 准备基础数据 (处理 ID 映射)
    # ---------------------------------------------------------
    
    # 1. 获取排序后的真实工人和批次ID
    # config_seru.worker_map: {logic_id: real_id}
    # config_seru.selected_batches: [real_id, real_id, ...]
    
    real_worker_ids = sorted([config_seru.worker_map[logic_id] for logic_id in config_seru.worker_map])
    worker_real_to_idx = {uid: i for i, uid in enumerate(real_worker_ids)} # 真实ID -> 矩阵索引(0~N)
    
    real_batch_ids = sorted(config_seru.selected_batches)
    batch_real_to_idx = {bid: i for i, bid in enumerate(real_batch_ids)}   # 真实ID -> 矩阵索引(0~M)

    # 2. 动态获取所有产品类型 (因为 ExcelDataLoader 中没有直接存储 product_types 列表)
    # 从 worker_to_product_dict 中提取所有出现过的 keys (Product IDs)
    all_product_types = set()
    if excel_loader.worker_to_product_dict:
        for prod_dict in excel_loader.worker_to_product_dict.values():
            all_product_types.update(prod_dict.keys())
    
    # 如果工人没涵盖所有产品，再检查一下批次字典 (双重保险)
    if excel_loader.batch_to_product_dict:
        for val in excel_loader.batch_to_product_dict.values():
            if '产品类型' in val:
                all_product_types.add(val['产品类型'])
                
    sorted_product_types = sorted(list(all_product_types))

    data_list = []

    # ---------------------------------------------------------
    # 1. Sheet: workerFeatureOriginal
    # ---------------------------------------------------------
    # Header: WorkerID + [Prod_1, Prod_2...] + Coeff
    header_worker = ['WorkerID'] + [f'Prod_{pt}' for pt in sorted_product_types] + ['Coeff']
    worker_data = [header_worker]

    for real_wid in real_worker_ids:
        row = [real_wid]
        
        # 获取该工人对每个产品的熟练度
        # 结构: worker_to_product_dict[worker_id][product_id] = efficiency
        worker_prods = excel_loader.worker_to_product_dict.get(real_wid, {})
        for pt in sorted_product_types:
            eff = worker_prods.get(pt, 0.0) # 如果没有该产品的熟练度，默认为0
            row.append(eff)
        
        # 获取多能工系数
        # 结构: worker_to_task_dict[worker_id]['系数'] = value
        coeff = 0.0
        task_info = excel_loader.worker_to_task_dict.get(real_wid)
        if task_info and '系数' in task_info:
            coeff = task_info['系数']
        row.append(coeff)
        
        worker_data.append(row)

    data_list.append({
        'SheetName': 'workerFeatureOriginal',
        'Data': worker_data
    })

    # ---------------------------------------------------------
    # 2. Sheet: batchInfo
    # ---------------------------------------------------------
    batch_data = [['BatchID', 'Type', 'size']]
    for real_bid in real_batch_ids:
        b_type = 0
        b_size = 0
        
        # 结构: batch_to_product_dict[batch_id] = {'产品类型': int, '批次大小': int}
        b_info = excel_loader.batch_to_product_dict.get(real_bid)
        if b_info:
            b_type = b_info.get('产品类型', 0)
            b_size = b_info.get('批次大小', 0)
            
        batch_data.append([real_bid, b_type, b_size])

    data_list.append({
        'SheetName': 'batchInfo',
        'Data': batch_data
    })
    # =========================================================
    # [新增] Sheet: workerFeature (归一化后的处理时间)
    # =========================================================
    # 1. 计算原始时间矩阵并找到最大值
    raw_times_matrix = []
    global_max_time = 0.0
    
    # 检查是否使用标准逻辑 (影响计算公式)
    use_standard_logic = hasattr(config_seru, 'use_standard_logic') and config_seru.use_standard_logic
    task_time_const = getattr(config_seru, 'task_time', 1.0) # 默认1.0

    for real_wid in real_worker_ids:
        row_times = []
        
        # 获取工人系数 'c'
        task_info = excel_loader.worker_to_task_dict.get(real_wid, {})
        worker_coeff_mult = task_info.get('系数', 0.0)
        
        c = 1.0
        if (config_seru.num_of_workers - config_seru.max_num_of_multiple_task) > 0:
            c += worker_coeff_mult * (config_seru.num_of_workers - config_seru.max_num_of_multiple_task)
            
        worker_prods = excel_loader.worker_to_product_dict.get(real_wid, {})

        for real_bid in real_batch_ids:
            # 获取批次信息
            b_info = excel_loader.batch_to_product_dict.get(real_bid)
            if not b_info:
                row_times.append(0.0)
                continue
            
            p_type = b_info.get('产品类型')
            b_size = b_info.get('批次大小', 0)
            
            # 工人对该产品的熟练度
            worker_prod_coeff = worker_prods.get(p_type, 0.0)
            
            # 计算单位任务时间 (task_time_in_seru)
            # 假设 Seru 只有该工人，所以 len(workers_set) = 1，不需要除以人数
            if use_standard_logic:
                unit_time = task_time_const * c * worker_prod_coeff
            else:
                unit_time = c * worker_prod_coeff
            
            # 计算总流转时间
            # 公式参考 calculate_fitness.py: time * batch_size * N / len(workers)
            # 这里 len(workers) = 1
            flow_time = unit_time * b_size * config_seru.num_of_workers
            
            row_times.append(flow_time)
            
            # 更新最大值
            if flow_time > global_max_time:
                global_max_time = flow_time
        
        raw_times_matrix.append(row_times)
    
    # 2. 归一化并生成表格数据
    # Header: WorkerID, B_101, B_205, ...
    wf_header = ['WorkerID'] + [f'B_{bid}' for bid in real_batch_ids]
    wf_data = [wf_header]
    
    for i, real_wid in enumerate(real_worker_ids):
        norm_row = [real_wid]
        raw_row = raw_times_matrix[i]
        for val in raw_row:
            if global_max_time > 1e-9: # 避免除以0
                norm_row.append(val / global_max_time)
            else:
                norm_row.append(0.0)
        wf_data.append(norm_row)

    data_list.append({
        'SheetName': 'workerFeature',
        'Data': wf_data
    })
    # =========================================================
    # 3. Sheet: result_value
    # ---------------------------------------------------------
    # 构建 batch 到 seru 的映射字典 (真实BatchID -> Seru序号 1-based)
    batch_to_seru_map = {}
    
    # 构建 SeruConfig 字符串 (使用真实ID)
    seru_config_list = []
    
    # 遍历当前解的 Seru 集合
    for idx, seru in enumerate(solution.formation.seru_set):
        seru_no = idx + 1 # Seru 编号从1开始
        
        # 将逻辑工人ID转为真实ID
        real_members = sorted([config_seru.worker_map.get(w, w) for w in seru.workers_set])
        seru_config_list.append(real_members)
        
        # 记录批次归属 (seru.batches_set 里存的已经是 真实ID，因为 calculate_fitness_edd 中使用了 selected_batches)
        # 但为了保险，还是确认一下数据流。如果之前 logic 改对了，这里就是 real_id。
        for batch_id in seru.batches_set:
            batch_to_seru_map[batch_id] = seru_no

    seru_config_str = f"SeruConfig{{serus={seru_config_list}}}"
    
    # 构建 result_value 的一行数据
    # Header: idx, SeruConfig, Cmax, 0 (start_flag), batch1_seru, batch2_seru...
    # 这里的 Batch 顺序必须严格对应 batchInfo 中的顺序 (即 real_batch_ids)
    
    result_header = ['idx', 'SeruConfig', 'Cmax', 'batchtoSeru_Start0'] + [f'B_{bid}' for bid in real_batch_ids]
    
    result_row = [
        run_idx, 
        seru_config_str, 
        solution.makespan,
        0 # 标志位
    ]
    
    for real_bid in real_batch_ids:
        # 获取该批次被分配到了哪个 Seru，未分配填 -1
        result_row.append(batch_to_seru_map.get(real_bid, -1))

    data_list.append({
        'SheetName': 'result_value',
        'Data': [result_header, result_row]
    })

    # ---------------------------------------------------------
    # 4. Sheet: Labels_worker (Adjacency Matrix)
    # ---------------------------------------------------------
    # 初始化 N x N 全0矩阵
    n_workers = len(real_worker_ids)
    w_matrix = [[0] * n_workers for _ in range(n_workers)]
    
    # 填充矩阵：同一个 Seru 内的工人两两标记为 1
    for seru in solution.formation.seru_set:
        # 获取该 Seru 中所有工人的 矩阵索引
        member_indices = []
        for logic_w in seru.workers_set:
            real_w = config_seru.worker_map.get(logic_w, logic_w)
            if real_w in worker_real_to_idx:
                member_indices.append(worker_real_to_idx[real_w])
        
        # 两两标记
        for i in member_indices:
            for j in member_indices:
                w_matrix[i][j] = 1 # 对角线也设为1
                # 如果不需要对角线为1，改为 if i != j: w_matrix[i][j] = 1

    w_data = [['Labels']] # Header
    for row in w_matrix:
        w_data.append(row)

    data_list.append({
        'SheetName': 'Labels_worker',
        'Data': w_data
    })

    # ---------------------------------------------------------
    # 5. Sheet: Labels_batches (Adjacency Matrix)
    # ---------------------------------------------------------
    # 初始化 M x M 全0矩阵
    n_batches = len(real_batch_ids)
    b_matrix = [[0] * n_batches for _ in range(n_batches)]
    
    # 填充矩阵：同一个 Seru 内的批次两两标记为 1
    for seru in solution.formation.seru_set:
        # 获取该 Seru 中所有批次的 矩阵索引
        batch_indices = []
        for b_id in seru.batches_set:
            # 同样假设 seru.batches_set 中已经是真实ID
            if b_id in batch_real_to_idx:
                batch_indices.append(batch_real_to_idx[b_id])
                
        # 两两标记
        for i in batch_indices:
            for j in batch_indices:
                b_matrix[i][j] = 1
                
    b_data = [['Labels']] # Header
    for row in b_matrix:
        b_data.append(row)
        
    data_list.append({
        'SheetName': 'Labels_batches',
        'Data': b_data
    })

    return data_list

def get_init_seru():
    ConfigLoader.preload_all()
    config_seru = ConfigLoader.get_config('config_seru')
    all_real_workers = list(range(1, 41))   
    all_real_batches = list(range(1, 501))
    
    # 随机采样
    selected_workers = random.sample(all_real_workers, config_seru.num_of_workers)
    selected_batches = random.sample(all_real_batches, config_seru.num_of_batches)

    # 建立映射字典：逻辑ID(1~N) -> 真实ID(Random)
    # 注意：逻辑ID必须从1开始，因为你的 Initialization 代码是从1生成的
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
    return worker_map, batch_map

def data_generate(num_examples):
    num1 = 0
    num2 = 0
    win = []
    cmax1 ,cmax2 = [],[]
    for i in range(num_examples):
        print(f"生成第 {i+1} 个数据集")
        worker_map, batch_map = get_init_seru()
        # 初始化GA
        loader = ExcelDataLoader()
        ga = GA(worker_map, batch_map)
        loader = ExcelDataLoader()
        ga2 = GA2(worker_map, batch_map)
        # ccea = CCEA(worker_map, batch_map)
        
        # 运行GA
        print(f"运行第 {i+1} 个数据集的GA2")
        best_formation, best_scheduling, best_solution1,archive, config_seru = ga2.run()
        cmax1.append(best_solution1.fitness)
        print(f"运行第 {i+1} 个数据集的GA")
        best_formation, best_scheduling, best_solution2,archive, config_seru = ga.run()
        cmax2.append(best_solution2.fitness)
        win.append((best_solution2.fitness - best_solution1.fitness)/best_solution2.fitness)
        if best_solution1.fitness < best_solution2.fitness:
            num1 += 1
        else:
            num2 += 1
        print(f"num1： {num1} ，num2： {num2}")
        print(f"cmax_ga2: {cmax1}")
        print(f"cmax_ga: {cmax2}")
        print(f"win: {win}")
        # best_formation_ccea, best_scheduling_ccea, best_solution_ccea = ccea.run()

        result_data = generate_result_data_structure(best_solution, config_seru)
        save_serudata_to_excel(f'datasets/JCompany/W{config_seru.num_of_workers}_J{config_seru.num_of_batches}/{int(i/100)}/{i+1}', result_data)

data_generate(5000)

# [
#     {
#         'SheetName': 'workerFeatureOriginal',
#         'Data': [
#             ['WorkerID', 'Features'],  # Features是每个工人对每个批次的熟练度，最后一位是该工人的多能工系数
#             [2, 0.490740741, 0.618686869,  0.259311869],
#             [4, 0.555555556, 0.727272727,  0.333333333]
#         ]
#     },
#     {
#         'SheetName': 'batchInfo',
#         'Data': [
#             ['BatchID', 'Type', 'size'],
#             [1, 4, 54],
#             [12, 1, 49]
#         ]
#     },
#     {
#         'SheetName': 'result_value',
#         'Data': [
#             ['idx', 'SeruConfig', 'Cmax', 'batchtoSeru'],#idx是1，SeruConfig是最终的-seru配置，Cmax是最终的最大完成时间，batchtoSeru是最终的每个批次分配到的-seru。开始位置的0表示批次分配的起始位置
#             [1, 'SeruConfig{serus=[[1, 7], [2, 8], [3], [4], [5], [6]]}',2883.596774,0,3,2,6,2,2,3,3,4,1,5],
#         ]

#     }
#     {
#         'SheetName': 'Labels_worker',
#         'Data': [
#             ['Labels'], #Labels是每个工人对每个工人的标签，1表示在最终的-seru配置中，该工人对在同一个-seru中，0表示不在同一个-seru中
#             [0,0,0,0,0,0,1,0],
#             [0,0,0,0,0,0,0,1],
#             [0,0,0,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0],
#             [1,0,0,0,0,0,0,0],
#             [0,1,0,0,0,0,0,0]
#         ]
#     },
#     {
#         'SheetName': 'Labels_batches',
#         'Data': [
#             ['Labels'], #Labels是每个批次对每个批次的标签，1表示在最终的-seru配置中，该批次对在同一个-seru中，0表示不在同一个-seru中
#             [0,0,0,0,0,1,1,0,0,0],
#             [0,0,0,1,1,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0,0,0],
#             [0,1,0,0,1,0,0,0,0,0],
#             [0,1,0,1,0,0,0,0,0,0],
#             [1,0,0,0,0,0,1,0,0,0],
#             [1,0,0,0,0,1,0,0,0,0],
#             [0,0,0,0,0,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0,0,0]
#         ]
#     },
# ]