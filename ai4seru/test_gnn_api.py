from re import A
from tokenize import group
from metaheuristics.ga_formation_edd.ga_formation_v3 import GA
from metaheuristics.ga_formation_edd.ga_formation_v3_2 import GA2
from metaheuristics.ccea.ccea_baseline import CCEA as BaselineCCEA
from metaheuristics.ccea.ccea_eta_mixed_original import CCEA as EtaMixedOriginalCCEA
from metaheuristics.ccea.ccea_eta_mixed_refined import CCEA as EtaMixedRefinedCCEA
from utils.config_loader import ConfigLoader
from utils.call_gnn_api_seru import get_gnn_result
from problem.pure_seru.pure_seru_entities import Solution
import random
from utils.excel_utils import ExcelDataLoader
from typing import List, Dict, Any
from utils.polt.draw_v2 import plot_a_b_matrices, plot_multi_matrices, plot_multi_xy_groups, plot_multi_xy_groups_minlen
from utils.excel_output import save_serudata_to_excel
from scipy import stats
import scikit_posthocs as sp
import numpy as np
import pandas as pd
import os
import copy  # 必须引入，用于深拷贝初始种群
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.initialization import Initialization

# ==============================================================================
# 数据预处理与初始化构建函数
# ==============================================================================
def build_payload(worker_map, batch_map, config_seru):
    loader = ExcelDataLoader.instance()
    loader.read_data(excel_path=config_seru.seru_data_path, config_sheet=config_seru)
    loader.read_data(excel_path=config_seru.due_dates_path, config_sheet=config_seru)
    loader.read_data(excel_path=config_seru.batch_types_path, config_sheet=config_seru)

    needed_product_types = set()
    for logic_bid in sorted(batch_map.keys()):
        real_bid = batch_map[logic_bid]
        binfo = loader.batch_to_product_dict[real_bid]  
        needed_product_types.add(str(binfo["产品类型"]))

    worker_to_task_dict = {}
    worker_to_product_dict = {}

    for logic_wid in sorted(worker_map.keys()):
        real_wid = worker_map[logic_wid]
        coeff = loader.worker_to_task_dict[real_wid]["系数"]
        worker_to_task_dict[str(logic_wid)] = {"系数": float(coeff)}

        real_prod_map = loader.worker_to_product_dict.get(real_wid, {})
        logic_prod_map = {}
        for pt in needed_product_types:
            val = real_prod_map.get(pt, None)
            if val is None and pt.isdigit():
                val = real_prod_map.get(int(pt), None)
            logic_prod_map[pt] = float(val) if val is not None else 0.0

        worker_to_product_dict[str(logic_wid)] = logic_prod_map

    batch_to_product_dict = {}
    for logic_bid in sorted(batch_map.keys()):
        real_bid = batch_map[logic_bid]
        binfo = loader.batch_to_product_dict[real_bid]
        batch_to_product_dict[str(logic_bid)] = {
            "产品类型": str(binfo["产品类型"]),
            "批次大小": int(binfo["批次大小"]),
        }

    payload = {
        "config_seru": {
            "num_of_workers": int(config_seru.num_of_workers),
            "num_of_batches": int(config_seru.num_of_batches),
            "max_num_of_multiple_task": int(config_seru.max_num_of_multiple_task),
        },
        "problem_data": {
            "worker_to_task_dict": worker_to_task_dict,
            "worker_to_product_dict": worker_to_product_dict,
            "batch_to_product_dict": batch_to_product_dict,
        }
    }
    return payload

def get_init_seru(config_seru):
    random.seed(42)
    all_real_workers = list(range(1, 41))   
    all_real_batches = list(range(1, 501))

    selected_workers = random.sample(all_real_workers, config_seru.num_of_workers)
    selected_batches = random.sample(all_real_batches, config_seru.num_of_batches)

    worker_map = {logic_id + 1: real_id for logic_id, real_id in enumerate(selected_workers)}
    batch_map = {logic_id + 1: real_id for logic_id, real_id in enumerate(selected_batches)}

    payload = build_payload(worker_map, batch_map, config_seru)
    print(f"随机映射工人: {worker_map}")
    print(f"随机选取批次: {batch_map}")
    return worker_map, batch_map, payload

def init_PS(N_W, N_B, pop_size):
    PSF = []
    for _ in range(pop_size):
        f_code = Initialization.initial_formation_code(N_W)
        f = SeruFormation(formation_code=f_code)
        Initialization.produce_seru_formation(N_W, f)
        PSF.append(f)

    PSS = []
    for _ in range(pop_size):
        s_code = Initialization.initial_schedule_code(N_W, N_B)
        s = SeruSchedule(schedule_code=s_code)
        Initialization.produce_seru_schedule(N_B, s)
        PSS.append(s)
    return PSF, PSS

# ==============================================================================
# 轨迹分析与截断辅助函数
# ==============================================================================
def get_cmax_at_t(history, t):
    best_cmax = float('inf')
    if not history: return best_cmax
    for record in history:
        record_cmax, record_time = record[0], record[1]
        if record_time <= t:
            best_cmax = min(best_cmax, record_cmax)
        else: break
    if best_cmax == float('inf') and len(history) > 0:
        best_cmax = history[0][0]
    return best_cmax

def get_time_to_target(history, target_cmax, max_time_default):
    for record in history:
        record_cmax, record_time = record[0], record[1]
        if record_cmax <= target_cmax + 1e-6: return record_time
    return max_time_default
def build_mixed_time_grid(max_sec=100.0):
    """
    构造分段采样时间网格：
    0~10s:   0.1s
    10~20s:  0.5s
    20s以后: 1.0s
    """
    times = []

    # [0, 10]
    t = 0.0
    while t <= min(10.0, max_sec) + 1e-9:
        times.append(round(t, 1))
        t += 0.1

    # (10, 20]
    if max_sec > 10.0:
        t = 10.5
        while t <= min(20.0, max_sec) + 1e-9:
            times.append(round(t, 1))
            t += 0.5

    # (20, max_sec]
    if max_sec > 20.0:
        t = 21.0
        while t <= max_sec + 1e-9:
            times.append(round(t, 1))
            t += 1.0

    return times


def get_trajectory_by_time_grid(history, time_grid):
    traj = []
    for t in time_grid:
        traj.append(get_cmax_at_t(history, t))
    return traj

# def get_per_second_trajectory(history, max_sec=100):
#     traj = []
#     for s in range(0, int(max_sec + 1)):
#         traj.append(get_cmax_at_t(history, s))
#     return traj

def simulate_early_stop(full_history, stop_val):
    """
    核心：通过事后截断，模拟较小的 iteration_stop 行为。
    必须确保传入的 full_history 是每一代 [cmax, time] 的连续记录。
    """
    if not full_history: return []
    truncated_history = []
    best_cmax = float('inf')
    no_improve_count = 0
    
    for record in full_history:
        current_cmax = record[0]
        truncated_history.append(record)
        
        # 检查是否有实质性进步 (加入 1e-6 容差防止浮点精度误判)
        if current_cmax < best_cmax - 1e-6:
            best_cmax = current_cmax
            no_improve_count = 0  # 有进步，计数器清零
        else:
            no_improve_count += 1 # 无进步，计数器累加
            
        # 连续未进步代数达到 stop_val，触发提前终止机制
        if no_improve_count >= stop_val:
            break
            
    return truncated_history


def truncate_history_by_time(history, t):
    truncated = [record for record in history if record[1] <= t]
    return truncated

def get_final_cmax_under_budget(history, t):
    truncated = truncate_history_by_time(history, t)
    if truncated:
        return truncated[-1][0]
    return history[0][0] if history else float('inf')

def get_time_to_target_with_budget(history, target_cmax, budget_t):
    for record in history:
        record_cmax, record_time = record[0], record[1]
        if record_time > budget_t:
            break
        if record_cmax <= target_cmax + 1e-6:
            return record_time
    return budget_t

# ==============================================================================
# 主生成与分析函数
# ==============================================================================
def data_generate(num_examples, W, J):
    ConfigLoader.preload_all()
    config_seru = ConfigLoader.get_config('config_seru')
    config_seru.num_of_workers = W
    config_seru.num_of_batches = J

    # iteration_stop thresholds under test
    iteration_stops = [5, 10, 20, 50]
    max_stop = max(iteration_stops)
    stop_time = 0.2*W*J

    # 嵌套字典结构：histories[stop_val]['CCEA1'] = [ run1_his, run2_his... ]
    histories = {
        stop: {'CCEA1': [], 'CCEA2': [], 'CCEA5': []} 
        for stop in iteration_stops
    }

    worker_map, batch_map, payload = get_init_seru(config_seru)
    
    for i in range(num_examples):
        print(f"\n========== 开始第 {i+1}/{num_examples} 次测试 (W={W}, J={J}) ==========")
        
        gnn_result = get_gnn_result(payload)
        edge_scores_worker = gnn_result["edge_scores_worker"]
        edge_scores_batch  = gnn_result["edge_scores_batch"]

        # 始终用同一套种群作对比，保证起点公平性
        PSF_50, PSS_50 = init_PS(config_seru.num_of_workers, config_seru.num_of_batches, 50)
        
        ccea1 = BaselineCCEA(worker_map=worker_map, batch_map=batch_map,
                      PSF=copy.deepcopy(PSF_50), PSS=copy.deepcopy(PSS_50))
                      
        ccea5 = EtaMixedOriginalCCEA(worker_map=worker_map, batch_map=batch_map,
                      edge_scores_worker=edge_scores_worker, edge_scores_batch=edge_scores_batch,
                      PSF=copy.deepcopy(PSF_50), PSS=copy.deepcopy(PSS_50))

        ccea2 = EtaMixedRefinedCCEA(worker_map=worker_map, batch_map=batch_map,
                      edge_scores_worker=edge_scores_worker, edge_scores_batch=edge_scores_batch,
                      PSF=copy.deepcopy(PSF_50), PSS=copy.deepcopy(PSS_50))
        
        # 1. Run once with the maximum stop value.
        print(f"--> 运行不含GNN算法 (CCEA1) [Max Stop = {max_stop}] ...")
        _, _, _, cmax_his1_full, _, _, _ = ccea1.run(Pop_Size=50, iteration_stop=max_stop, stop_time=stop_time)
        
        print(f"--> 运行含GNN算法 (CCEA2) [Max Stop = {max_stop}] ...")
        _, _, _, cmax_his2_full, _, _, _ = ccea2.run(Pop_Size=50, iteration_stop=max_stop, stop_time=stop_time)

        print(f"--> 运行含GNN算法 (CCEA5) [Max Stop = {max_stop}] ...")
        _, _, _, cmax_his5_full, _, _, _ = ccea5.run(Pop_Size=50, iteration_stop=max_stop, stop_time=stop_time)
        
        # 2. 轨迹切片：模拟不同提前终止策略的效果
        for stop_val in iteration_stops:
            histories[stop_val]['CCEA1'].append(simulate_early_stop(cmax_his1_full, stop_val))
            histories[stop_val]['CCEA2'].append(simulate_early_stop(cmax_his2_full, stop_val))
            histories[stop_val]['CCEA5'].append(simulate_early_stop(cmax_his5_full, stop_val))

    print("\n测试运行结束，开始分析多组截面数据并导出Excel...")

    # ==============================================================================
    # 提取离散时间点截面指标并汇总 (Summary Sheets)
    # ==============================================================================
    # target_times = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,
    #                 110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,
    #                 260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,
    #                 410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,
    #                 560,570,580,590,600]
    # target_times = W * J * alpha
   # ==============================================================================
# 提取离散时间点截面指标并汇总 (Summary Sheets)
# ==============================================================================
    # 用字典绑定绝对时间 t 和 对应的 alpha_t
    time_alpha_map = {}
    
    # 0-10 部分的 alpha_t 设为 0
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        time_alpha_map[t] = 0.0
        
    # alpha_t 列表部分
    alpha_list = [0.01, 0.015, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    for a in alpha_list:
        t_val = W * J * a
        time_alpha_map[t_val] = a  # Exact time collisions retain the observed alpha value.
        
    # 将字典按时间 t 从小到大排序，生成 (t, alpha) 的元组列表
    # sorted_time_items = sorted(time_alpha_map.items(), key=lambda x: x[0])

    max_record_sec = stop_time
    summary_data = []

    for stop_val in iteration_stops:
        # --------------------------------------------------------------------------
        # 4) 再逐个时间截面 t 统计
        # --------------------------------------------------------------------------
        for t, current_alpha in time_alpha_map.items():
            cmax_c1_t, cmax_c5_t, cmax_c2_t = [], [], []
            
            final_c1_at_t_runs = []
            final_c5_at_t_runs = []
            final_c2_at_t_runs = []

            his1_list_t = []
            his5_list_t = []
            his2_list_t = []

            for idx in range(num_examples):
                his1 = histories[stop_val]['CCEA1'][idx]
                his5 = histories[stop_val]['CCEA5'][idx]
                his2 = histories[stop_val]['CCEA2'][idx]

                his1_list_t.append(his1)
                his5_list_t.append(his5)
                his2_list_t.append(his2)

                # 当前截面 t 的 cmax
                c1 = get_cmax_at_t(his1, t)
                c5 = get_cmax_at_t(his5, t)
                c2 = get_cmax_at_t(his2, t)

                cmax_c1_t.append(c1)
                cmax_c5_t.append(c5)
                cmax_c2_t.append(c2)

                # 当前 budget=t 下，每条轨迹的“最终 cmax”
                final_c1_at_t_runs.append(get_final_cmax_under_budget(his1, t))
                final_c5_at_t_runs.append(get_final_cmax_under_budget(his5, t))
                final_c2_at_t_runs.append(get_final_cmax_under_budget(his2, t))

            # ======================================================================
            # 1. 基础均值与【样本标准差(ddof=1)】计算
            # ======================================================================
            mean_c1 = np.mean(cmax_c1_t)
            std_c1 = np.std(cmax_c1_t, ddof=1) if len(cmax_c1_t) > 1 else 0
            
            mean_c5 = np.mean(cmax_c5_t)
            std_c5 = np.std(cmax_c5_t, ddof=1) if len(cmax_c5_t) > 1 else 0
            
            mean_c2 = np.mean(cmax_c2_t)
            std_c2 = np.std(cmax_c2_t, ddof=1) if len(cmax_c2_t) > 1 else 0

            # 当前 budget=t 下，各算法的最终平均 cmax
            final_mean_c1_at_t = np.mean(final_c1_at_t_runs)
            final_mean_c5_at_t = np.mean(final_c5_at_t_runs)
            final_mean_c2_at_t = np.mean(final_c2_at_t_runs)

            # ======================================================================
            # 2. Independently paired Gap, Speedup, and Saving Ratio
            # ======================================================================
            cmax_gap5_runs, cmax_gap2_runs = [], []
            speedup_mixed_runs, speedup_mixed2_runs = [], []
            saving_mixed_runs, saving_mixed2_runs = [], []
            
            time_c1_self_reach_runs = []
            time_c5_reach_c1_runs = []
            time_c2_reach_c1_runs = []
            time_c5_self_reach_runs = []
            time_c2_self_reach_runs = []

            # 定义最小时间分辨率（1毫秒），彻底根除除零错误和 0/0 打平错误
            MIN_TIME_RES = 1e-3 

            for idx in range(num_examples):
                # ---- 配对 Cmax Gap ----
                c1, c5, c2 = cmax_c1_t[idx], cmax_c5_t[idx], cmax_c2_t[idx]
                cmax_gap5_runs.append((c1 - c5) / c1 if c1 > 0 else 0.0)
                cmax_gap2_runs.append((c1 - c2) / c1 if c1 > 0 else 0.0)

                # ---- 专属靶心 ----
                target_cmax_idx_c1 = final_c1_at_t_runs[idx]
                target_cmax_idx_c5 = final_c5_at_t_runs[idx]
                target_cmax_idx_c2 = final_c2_at_t_runs[idx]

                # ---- 到达时间计算 ----
                t_c1_self = get_time_to_target_with_budget(his1_list_t[idx], target_cmax_idx_c1, t)
                t_c5_self = get_time_to_target_with_budget(his5_list_t[idx], target_cmax_idx_c5, t)
                t_c2_self = get_time_to_target_with_budget(his2_list_t[idx], target_cmax_idx_c2, t)
                
                t_c5_final = get_time_to_target_with_budget(his5_list_t[idx], target_cmax_idx_c1, t)
                t_c2_final = get_time_to_target_with_budget(his2_list_t[idx], target_cmax_idx_c1, t)

                time_c1_self_reach_runs.append(t_c1_self)
                time_c5_self_reach_runs.append(t_c5_self)
                time_c2_self_reach_runs.append(t_c2_self)
                time_c5_reach_c1_runs.append(t_c5_final)
                time_c2_reach_c1_runs.append(t_c2_final)

                # ---- 配对加速比与配对节省率 (加入安全底线) ----
                t_c1_safe = max(t_c1_self, MIN_TIME_RES)
                t_c5_safe = max(t_c5_final, MIN_TIME_RES)
                t_c2_safe = max(t_c2_final, MIN_TIME_RES)

                speedup_mixed_runs.append(t_c1_safe / t_c5_safe)
                speedup_mixed2_runs.append(t_c1_safe / t_c2_safe)

                saving_mixed_runs.append((t_c1_safe - t_c5_safe) / t_c1_safe)
                saving_mixed2_runs.append((t_c1_safe - t_c2_safe) / t_c1_safe)

            # --- 汇总所有配对指标的均值 ---
            c_gap5 = np.mean(cmax_gap5_runs)
            c_gap2 = np.mean(cmax_gap2_runs)
            
            speedup_mixed_at_t = np.mean(speedup_mixed_runs)
            speedup_mixed2_at_t = np.mean(speedup_mixed2_runs)
            
            time_saving_ratio_t_mixed = np.mean(saving_mixed_runs)
            time_saving_ratio_t_mixed2 = np.mean(saving_mixed2_runs)

            # --- 各类到达时间的均值 ---
            mean_time_c1_to_final_c1_at_t = np.mean(time_c1_self_reach_runs)
            mean_time_c5_to_final_c1_at_t = np.mean(time_c5_reach_c1_runs)
            mean_time_c2_to_final_c1_at_t = np.mean(time_c2_reach_c1_runs)
            
            mean_time_c5_to_final_c5_at_t = np.mean(time_c5_self_reach_runs)
            mean_time_c2_to_final_c2_at_t = np.mean(time_c2_self_reach_runs)

            # ======================================================================
            # 3. Wilcoxon 统计检验
            # ======================================================================
            try:
                stat_m5_two, p_m5_two = stats.wilcoxon(cmax_c5_t, cmax_c1_t, alternative="two-sided")
            except ValueError:
                stat_m5_two, p_m5_two = 0, 1.0

            try:
                stat_m5_less, p_m5_less = stats.wilcoxon(cmax_c5_t, cmax_c1_t, alternative="less")
            except ValueError:
                stat_m5_less, p_m5_less = 0, 1.0

            try:
                stat_m2_two, p_m2_two = stats.wilcoxon(cmax_c2_t, cmax_c1_t, alternative="two-sided")
            except ValueError:
                stat_m2_two, p_m2_two = 0, 1.0

            try:
                stat_m2_less, p_m2_less = stats.wilcoxon(cmax_c2_t, cmax_c1_t, alternative="less")
            except ValueError:
                stat_m2_less, p_m2_less = 0, 1.0

            # ======================================================================
            # 4. 汇总到 summary_data
            # ======================================================================
            summary_data.append({
                "Iteration_Stop": stop_val,
                "Alpha_t": current_alpha,
                "Time(s)": t,

                "Cmax_CCEA(No GNN)": mean_c1,
                "Std_CCEA": std_c1,

                "Cmax_GNN(Mixed)": mean_c5,
                "Std_GNN(Mixed)": std_c5,
                "Cmax_Gap(Mixed)": c_gap5,
                "p_two_sided(Mixed_vs_NoGNN)": p_m5_two,
                "p_one_sided_less(Mixed_vs_NoGNN)": p_m5_less,

                "Cmax_GNN(Mixed)2": mean_c2,
                "Std_GNN(Mixed)2": std_c2,
                "Cmax_Gap(Mixed2)": c_gap2,
                "p_two_sided(Mixed2_vs_NoGNN)": p_m2_two,
                "p_one_sided_less(Mixed2_vs_NoGNN)": p_m2_less,

                # 截面时间指标：相对 baseline 的时间节省比例 (基于配对计算)
                "Time_Saving_Ratio_at_t(Mixed)": time_saving_ratio_t_mixed,
                "Time_Saving_Ratio_at_t(Mixed2)": time_saving_ratio_t_mixed2,

                # 最终平均收敛值
                "Mean_Final_Cmax_CCEA(No GNN)_at_t": final_mean_c1_at_t,
                "Mean_Final_Cmax_GNN(Mixed)_at_t": final_mean_c5_at_t,
                "Mean_Final_Cmax_GNN(Mixed)2_at_t": final_mean_c2_at_t,

                "Time_NoGNN_to_Final_Cmax_of_NoGNN_at_t": mean_time_c1_to_final_c1_at_t,
                "Time_Mixed_to_Final_Cmax_of_Mixed_at_t": mean_time_c5_to_final_c5_at_t,
                "Time_Mixed2_to_Final_Cmax_of_Mixed2_at_t": mean_time_c2_to_final_c2_at_t,

                "Time_Mixed_to_Final_Cmax_of_NoGNN_at_t": mean_time_c5_to_final_c1_at_t,
                "Time_Mixed2_to_Final_Cmax_of_NoGNN_at_t": mean_time_c2_to_final_c1_at_t,

                "Speedup_Mixed_vs_NoGNN_at_t": speedup_mixed_at_t,
                "Speedup_Mixed2_vs_NoGNN_at_t": speedup_mixed2_at_t,
            })

    df_summary = pd.DataFrame(summary_data)

    # ==============================================================================
    # 提取完整迭代轨迹 (Full Trajectory Sheets)
    # ==============================================================================
    per_sec_dfs = {}

    # Stopping points included in the trajectory export
    trajectory_stops_to_export = [20, 50]

    # 构造混合时间网格
    time_grid = build_mixed_time_grid(max_record_sec)

    # Excel列名
    time_cols = ["Run_ID"] + [
        f"{t:.1f}s" if abs(t - round(t)) > 1e-9 else f"{int(round(t))}s"
        for t in time_grid
    ]

    for stop_val in trajectory_stops_to_export:
        for algo in ['CCEA1', 'CCEA2', 'CCEA5']:
            sec_data = []

            for idx in range(num_examples):
                full_history = histories[stop_val][algo][idx]
                traj = get_trajectory_by_time_grid(full_history, time_grid)
                sec_data.append([f"Run_{idx+1}"] + traj)
            
            # cols = ["Run_ID"] + [f"{s}s" for s in range(0, int(max_record_sec + 1))]
            df_sec = pd.DataFrame(sec_data, columns=time_cols)
            
            mean_values = df_sec.iloc[:, 1:].mean().tolist()
            std_values = df_sec.iloc[:, 1:].std().tolist()  # The Mean row is not present yet.

            df_sec.loc[len(df_sec)] = ["Mean"] + mean_values
            df_sec.loc[len(df_sec)] = ["Std"] + std_values
            # df_sec.loc[len(df_sec)] = ["Std"] + df_sec.iloc[:, 1:-1].std().tolist() + [df_sec.iloc[:, -1].std()]
            
            per_sec_dfs[f"Traj_Stop_{stop_val}_{algo}"] = df_sec

    # ==============================================================================
    # 输出保存到同一个 Excel (精准生成 5 + 3 = 8 个 Sheet)
    # ==============================================================================
    path = "outputsCon260323"
    os.makedirs(path, exist_ok=True)
    excel_filename = path+f"/W{W}_J{J}_GNN_MultiStop_Result.xlsx"
    
    with pd.ExcelWriter(excel_filename) as writer:
        
        # 1. 保存 5 个不同 iteration_stop 的综合结果
        for stop_val in iteration_stops:
            df_sub = df_summary[df_summary['Iteration_Stop'] == stop_val]
            df_sub.to_excel(writer, sheet_name=f"Summary_Stop_{stop_val}", index=False)
            
        # 2. 保存 3 个算法的最全迭代轨迹
        for sheet_name, df_sec in per_sec_dfs.items():
            safe_sheet_name = sheet_name[:31]
            df_sec.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        
    print(f"\n数据处理完毕！成功生成 10 个 Sheet 的结果文件：{excel_filename}")

    # ==============================================================================
    # 绘图逻辑保持原有习惯
    # ==============================================================================
    for stop_val in iteration_stops:
        plot_multi_xy_groups(
            {
                "GNN_Mixed(CCEA5)": histories[stop_val]['CCEA5'], 
                "NO_GNN(CCEA1)": histories[stop_val]['CCEA1']
            },
            save_dir=path,
            prefix=f"W{W}_J{J}_Stop{stop_val}_compare",
            ci="stderr",
            grid="union",
            max_runs_plot=20,
            truncate="min_time",
        )

# 执行入口
if __name__ == "__main__":
    W = [15,25]
    J = [50,100, 150, 200, 300]
    for w in W:
        for j in J:
            data_generate(20, w, j)
