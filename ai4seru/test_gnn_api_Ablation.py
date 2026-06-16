from metaheuristics.ga_formation_edd.ga_formation_v3 import GA
from metaheuristics.ga_formation_edd.ga_formation_v3_2 import GA2
from metaheuristics.ccea.ccea_conflict_minimal_v4_hotstart_v2 import CCEA as CCEA4
from metaheuristics.ccea.ccea_conflict_minimal_v4_hotstart_v3_fixed import CCEA as CCEA3
from metaheuristics.ccea.ccea_conflict_minimal_v4_hotstart_v3_mixed import CCEA as CCEA2
from metaheuristics.ccea.ccea_conflict_minimal_v4_hotstart_v3_mixed_fixed import CCEA as CCEA5
from metaheuristics.ccea.ccea_conflict_minimal_v4_hotstart_v3_mixed_fixed_2_ablation import CCEA as CCEA6
from metaheuristics.ccea.ccea import CCEA
from utils.config_loader import ConfigLoader
from utils.call_gnn_api_seru import get_gnn_result
from utils.excel_utils import ExcelDataLoader
from utils.polt.draw_v2 import plot_multi_xy_groups
from scipy import stats
import numpy as np
import pandas as pd
import random
import os
import copy
import matplotlib.pyplot as plt
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule
from problem.pure_seru.initialization import Initialization


# ==============================================================================
# 数据预处理与初始化构建函数
# ==============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)



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



def get_init_seru(config_seru, seed=42):
    random.seed(seed)
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
    if not history:
        return best_cmax
    for record in history:
        record_cmax, record_time = record[0], record[1]
        if record_time <= t:
            best_cmax = min(best_cmax, record_cmax)
        else:
            break
    if best_cmax == float('inf') and len(history) > 0:
        best_cmax = history[0][0]
    return best_cmax



def build_mixed_time_grid(max_sec=100.0):
    times = []

    t = 0.0
    while t <= min(10.0, max_sec) + 1e-9:
        times.append(round(t, 1))
        t += 0.1

    if max_sec > 10.0:
        t = 10.5
        while t <= min(20.0, max_sec) + 1e-9:
            times.append(round(t, 1))
            t += 0.5

    if max_sec > 20.0:
        t = 21.0
        while t <= max_sec + 1e-9:
            times.append(round(t, 1))
            t += 1.0

    return times



def get_trajectory_by_time_grid(history, time_grid):
    return [get_cmax_at_t(history, t) for t in time_grid]



def simulate_early_stop(full_history, stop_val):
    if not full_history:
        return []
    truncated_history = []
    best_cmax = float('inf')
    no_improve_count = 0

    for record in full_history:
        current_cmax = record[0]
        truncated_history.append(record)

        if current_cmax < best_cmax - 1e-6:
            best_cmax = current_cmax
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= stop_val:
            break

    return truncated_history



def truncate_history_by_time(history, t):
    return [record for record in history if record[1] <= t]



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



def get_final_cmax(history):
    return history[-1][0] if history else float('inf')



def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else float('nan')



def safe_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0



def safe_pct_improvement(base_value, new_value):
    if abs(base_value) <= 1e-12:
        return 0.0
    return (base_value - new_value) / base_value * 100.0



def wilcoxon_pvalue(x, y, alternative="two-sided"):
    try:
        _, p = stats.wilcoxon(x, y, alternative=alternative)
        return float(p)
    except ValueError:
        return 1.0



def classify_paired_result(variant_values, baseline_values):
    mean_variant = safe_mean(variant_values)
    mean_baseline = safe_mean(baseline_values)
    p_less = wilcoxon_pvalue(variant_values, baseline_values, alternative="less")
    p_greater = wilcoxon_pvalue(variant_values, baseline_values, alternative="greater")

    if mean_variant < mean_baseline - 1e-12 and p_less < 0.05:
        return "+"
    if mean_variant > mean_baseline + 1e-12 and p_greater < 0.05:
        return "-"
    return "="



def count_wtl(variant_values, baseline_values, tol=1e-12):
    wins = 0
    ties = 0
    losses = 0
    for v, b in zip(variant_values, baseline_values):
        if v < b - tol:
            wins += 1
        elif v > b + tol:
            losses += 1
        else:
            ties += 1
    return f"{wins}/{ties}/{losses}"



def flatten_runs(nested_histories):
    flat = []
    for inst_runs in nested_histories:
        flat.extend(inst_runs)
    return flat



def make_histories_container(iteration_stops, num_instances):
    return {
        stop: {
            'CCEA1': [[] for _ in range(num_instances)],
            'CCEA2': [[] for _ in range(num_instances)],
            'CCEA3': [[] for _ in range(num_instances)],
            'CCEA5': [[] for _ in range(num_instances)],
        }
        for stop in iteration_stops
    }



def build_ablation_table1(histories, iteration_stops, W, J, instance_seeds, num_repeats):
    """
    表1：最终解质量（5个实例 × 每实例20次）
    统计逻辑：先对每个实例内部20次重复求均值，再在5个实例上汇总。
    """
    algo_map = {
        "CCEA": "CCEA1",
        "ETA-F": "CCEA2",
        "ETA-S": "CCEA3",
        "ETA-FS": "CCEA5",
    }

    rows = []
    instance_name = f"W{W}_J{J}"
    num_instances = len(instance_seeds)
    total_runs = num_instances * num_repeats

    for stop_val in iteration_stops:
        instance_level = {k: [] for k in algo_map.keys()}
        for inst_id in range(num_instances):
            for paper_name, raw_name in algo_map.items():
                finals = [get_final_cmax(h) for h in histories[stop_val][raw_name][inst_id]]
                instance_level[paper_name].append(safe_mean(finals))

        means = {k: safe_mean(v) for k, v in instance_level.items()}
        stds = {k: safe_std(v) for k, v in instance_level.items()}

        rows.append({
            "Instance_Group": instance_name,
            "Num_Instances": num_instances,
            "Repeats_Per_Instance": num_repeats,
            "Total_Runs": total_runs,
            "Iteration_Stop": stop_val,
            "CCEA_Mean_Final_Cmax": means["CCEA"],
            "CCEA_Std_Final_Cmax": stds["CCEA"],
            "ETA-F_Mean_Final_Cmax": means["ETA-F"],
            "ETA-F_Std_Final_Cmax": stds["ETA-F"],
            "ETA-S_Mean_Final_Cmax": means["ETA-S"],
            "ETA-S_Std_Final_Cmax": stds["ETA-S"],
            "ETA-FS_Mean_Final_Cmax": means["ETA-FS"],
            "ETA-FS_Std_Final_Cmax": stds["ETA-FS"],
            "Gap_ETA-F_vs_CCEA(%)": safe_pct_improvement(means["CCEA"], means["ETA-F"]),
            "Gap_ETA-S_vs_CCEA(%)": safe_pct_improvement(means["CCEA"], means["ETA-S"]),
            "Gap_ETA-FS_vs_CCEA(%)": safe_pct_improvement(means["CCEA"], means["ETA-FS"]),
            "Gap_CCEA_vs_ETA-FS(%)": safe_pct_improvement(means["CCEA"], means["ETA-FS"]),
            "Gap_ETA-F_vs_ETA-FS(%)": safe_pct_improvement(means["ETA-F"], means["ETA-FS"]),
            "Gap_ETA-S_vs_ETA-FS(%)": safe_pct_improvement(means["ETA-S"], means["ETA-FS"]),
        })

    return pd.DataFrame(rows)



def build_ablation_table2(histories, iteration_stops, stop_time, W, J, instance_seeds, num_repeats):
    """
    表2：加速效果 + 显著性（5个实例 × 每实例20次）
    统计逻辑：每个实例先基于其20次重复计算 instance-level mean time-to-target；
    再在5个实例层面做均值、Wilcoxon 和 W/T/L 汇总。
    目标值固定为“该实例下 CCEA 的平均最终 Cmax”。
    """
    algo_map = {
        "CCEA": "CCEA1",
        "ETA-F": "CCEA2",
        "ETA-S": "CCEA3",
        "ETA-FS": "CCEA5",
    }

    rows = []
    instance_name = f"W{W}_J{J}"
    num_instances = len(instance_seeds)
    total_runs = num_instances * num_repeats

    for stop_val in iteration_stops:
        target_cmax_by_instance = []
        instance_mean_times = {k: [] for k in algo_map.keys()}

        for inst_id in range(num_instances):
            ccea_finals = [get_final_cmax(h) for h in histories[stop_val][algo_map["CCEA"]][inst_id]]
            target_cmax = safe_mean(ccea_finals)
            target_cmax_by_instance.append(target_cmax)

            for paper_name, raw_name in algo_map.items():
                times = [
                    get_time_to_target_with_budget(h, target_cmax, stop_time)
                    for h in histories[stop_val][raw_name][inst_id]
                ]
                instance_mean_times[paper_name].append(safe_mean(times))

        row = {
            "Instance_Group": instance_name,
            "Num_Instances": num_instances,
            "Repeats_Per_Instance": num_repeats,
            "Total_Runs": total_runs,
            "Iteration_Stop": stop_val,
            "Target_Cmax(Mean_Final_CCEA)_Across_Instances": safe_mean(target_cmax_by_instance),
            "CCEA_Mean_Time_to_Target(s)": safe_mean(instance_mean_times["CCEA"]),
            "CCEA_Std_Time_to_Target(s)": safe_std(instance_mean_times["CCEA"]),
        }

        for variant in ["ETA-F", "ETA-S", "ETA-FS"]:
            row[f"{variant}_Mean_Time_to_Target(s)"] = safe_mean(instance_mean_times[variant])
            row[f"{variant}_Std_Time_to_Target(s)"] = safe_std(instance_mean_times[variant])
            row[f"{variant}_Time_Reduction_vs_CCEA(%)"] = safe_pct_improvement(
                safe_mean(instance_mean_times["CCEA"]),
                safe_mean(instance_mean_times[variant])
            )
            row[f"{variant}_p_time_less_vs_CCEA"] = wilcoxon_pvalue(
                instance_mean_times[variant], instance_mean_times["CCEA"], alternative="less"
            )
            row[f"{variant}_Result_vs_CCEA"] = classify_paired_result(
                instance_mean_times[variant], instance_mean_times["CCEA"]
            )
            row[f"{variant}_W/T/L_vs_CCEA"] = count_wtl(
                instance_mean_times[variant], instance_mean_times["CCEA"]
            )

        rows.append(row)

    return pd.DataFrame(rows)



def build_paper_figure_data(histories_one_stop_one_instance, max_sec):
    time_grid = build_mixed_time_grid(max_sec)
    algo_map = {
        "CCEA": "CCEA1",
        "ETA-F": "CCEA2",
        "ETA-S": "CCEA3",
        "ETA-FS": "CCEA5",
    }

    df = pd.DataFrame({"Time(s)": time_grid})
    for paper_name, raw_name in algo_map.items():
        traj_matrix = []
        for history in histories_one_stop_one_instance[raw_name]:
            traj_matrix.append(get_trajectory_by_time_grid(history, time_grid))
        traj_matrix = np.asarray(traj_matrix, dtype=float)

        if traj_matrix.size == 0:
            mean_vals = np.full(len(time_grid), np.nan)
            stderr_vals = np.full(len(time_grid), np.nan)
        else:
            mean_vals = np.mean(traj_matrix, axis=0)
            if traj_matrix.shape[0] > 1:
                stderr_vals = np.std(traj_matrix, axis=0, ddof=1) / np.sqrt(traj_matrix.shape[0])
            else:
                stderr_vals = np.zeros(traj_matrix.shape[1])

        df[f"{paper_name}_Mean_Cmax"] = mean_vals
        df[f"{paper_name}_StdErr_Cmax"] = stderr_vals

    return df



def plot_paper_ablation_figure(fig_df, save_path, W, J, stop_val, representative_instance_seed):
    plt.figure(figsize=(8, 5))
    algo_names = ["CCEA", "ETA-F", "ETA-S", "ETA-FS"]

    for algo_name in algo_names:
        x = fig_df["Time(s)"].to_numpy(dtype=float)
        y = fig_df[f"{algo_name}_Mean_Cmax"].to_numpy(dtype=float)
        yerr = fig_df[f"{algo_name}_StdErr_Cmax"].to_numpy(dtype=float)

        plt.plot(x, y, linewidth=2, label=algo_name)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.20)

    plt.xlabel("Time (s)")
    plt.ylabel(r"Best-so-far $C_{\max}$")
    plt.title(
        f"Convergence curves on representative instance W={W}, J={J} "
        f"(iteration_stop={stop_val}, seed={representative_instance_seed})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 主生成与分析函数
# ==============================================================================
def data_generate(num_repeats, W, J, seed=42, paper_fig_stop=50, num_instances=5):
    ConfigLoader.preload_all()
    config_seru = ConfigLoader.get_config('config_seru')
    config_seru.num_of_workers = W
    config_seru.num_of_batches = J

    iteration_stops = [5, 10, 20, 50]
    max_stop = max(iteration_stops)
    if paper_fig_stop not in iteration_stops:
        paper_fig_stop = max_stop

    alpha_list = [0.01, 0.015, 0.02, 0.05]
    stop_time = alpha_list[-1] * W * J
    instance_seeds = [seed + i for i in range(num_instances)]

    histories = make_histories_container(iteration_stops, num_instances)
    instance_infos = []

    for inst_id, instance_seed in enumerate(instance_seeds):
        print(f"\n{'=' * 20} 开始生成第 {inst_id + 1}/{num_instances} 个实例, instance_seed={instance_seed} {'=' * 20}")
        worker_map, batch_map, payload = get_init_seru(config_seru, seed=instance_seed)
        gnn_result = get_gnn_result(payload)
        edge_scores_worker = gnn_result["edge_scores_worker"]
        edge_scores_batch = gnn_result["edge_scores_batch"]

        instance_infos.append({
            "instance_id": inst_id + 1,
            "instance_seed": instance_seed,
            "worker_map": worker_map,
            "batch_map": batch_map,
        })

        for rep in range(num_repeats):
            run_seed = instance_seed * 100000 + rep
            print(
                f"\n---------- 实例 {inst_id + 1}/{num_instances}, 重复 {rep + 1}/{num_repeats} "
                f"(instance_seed={instance_seed}, run_seed={run_seed}) ----------"
            )

            seed_everything(run_seed)
            PSF_50, PSS_50 = init_PS(config_seru.num_of_workers, config_seru.num_of_batches, 50)

            ccea1 = CCEA4(
                worker_map=worker_map,
                batch_map=batch_map,
                PSF=copy.deepcopy(PSF_50),
                PSS=copy.deepcopy(PSS_50)
            )
            ccea5 = CCEA6(
                worker_map=worker_map,
                batch_map=batch_map,
                edge_scores_worker=edge_scores_worker,
                edge_scores_batch=edge_scores_batch,
                PSF=copy.deepcopy(PSF_50),
                PSS=copy.deepcopy(PSS_50)
            )
            ccea2 = CCEA6(
                worker_map=worker_map,
                batch_map=batch_map,
                edge_scores_worker=edge_scores_worker,
                PSF=copy.deepcopy(PSF_50),
                PSS=copy.deepcopy(PSS_50)
            )
            ccea3 = CCEA6(
                worker_map=worker_map,
                batch_map=batch_map,
                edge_scores_batch=edge_scores_batch,
                PSF=copy.deepcopy(PSF_50),
                PSS=copy.deepcopy(PSS_50)
            )

            seed_everything(run_seed)
            print(f"--> 运行不含GNN算法 (CCEA1) [Max Stop = {max_stop}] ...")
            _, _, _, cmax_his1_full, _, _, _ = ccea1.run(Pop_Size=50, iteration_stop=max_stop, stop_time=stop_time)

            seed_everything(run_seed)
            print(f"--> 运行含GNN算法 (worker only) [Max Stop = {max_stop}] ...")
            _, _, _, cmax_his2_full, _, _, _ = ccea2.run(Pop_Size=50, iteration_stop=max_stop, stop_time=stop_time)

            seed_everything(run_seed)
            print(f"--> 运行含GNN算法 (batch only) [Max Stop = {max_stop}] ...")
            _, _, _, cmax_his3_full, _, _, _ = ccea3.run(Pop_Size=50, iteration_stop=max_stop, stop_time=stop_time)

            seed_everything(run_seed)
            print(f"--> 运行含GNN算法 (both) [Max Stop = {max_stop}] ...")
            _, _, _, cmax_his5_full, _, _, _ = ccea5.run(Pop_Size=50, iteration_stop=max_stop, stop_time=stop_time)

            for stop_val in iteration_stops:
                histories[stop_val]['CCEA1'][inst_id].append(simulate_early_stop(cmax_his1_full, stop_val))
                histories[stop_val]['CCEA2'][inst_id].append(simulate_early_stop(cmax_his2_full, stop_val))
                histories[stop_val]['CCEA3'][inst_id].append(simulate_early_stop(cmax_his3_full, stop_val))
                histories[stop_val]['CCEA5'][inst_id].append(simulate_early_stop(cmax_his5_full, stop_val))

    print("\n测试运行结束，开始分析多组截面数据并导出Excel...")

    time_alpha_map = {}
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        time_alpha_map[t] = 0.0
    for a in alpha_list:
        t_val = W * J * a
        time_alpha_map[t_val] = a

    max_record_sec = stop_time
    summary_data = []
    total_runs = num_instances * num_repeats

    for stop_val in iteration_stops:
        flat_his1 = flatten_runs(histories[stop_val]['CCEA1'])
        flat_his2 = flatten_runs(histories[stop_val]['CCEA2'])
        flat_his3 = flatten_runs(histories[stop_val]['CCEA3'])
        flat_his5 = flatten_runs(histories[stop_val]['CCEA5'])

        for t, current_alpha in sorted(time_alpha_map.items(), key=lambda x: x[0]):
            cmax_c1_t, cmax_c5_t, cmax_c2_t, cmax_c3_t = [], [], [], []
            final_c1_at_t_runs, final_c5_at_t_runs = [], []
            final_c2_at_t_runs, final_c3_at_t_runs = [], []
            time_c1_self_reach_runs, time_c5_reach_c1_runs = [], []
            time_c2_reach_c1_runs, time_c3_reach_c1_runs = [], []
            time_c5_self_reach_runs, time_c2_self_reach_runs, time_c3_self_reach_runs = [], [], []
            cmax_gap5_runs, cmax_gap2_runs, cmax_gap3_runs = [], [], []
            speedup_mixed_runs, speedup_mixed2_runs, speedup_mixed3_runs = [], [], []
            saving_mixed_runs, saving_mixed2_runs, saving_mixed3_runs = [], [], []

            MIN_TIME_RES = 1e-3

            for idx in range(total_runs):
                his1 = flat_his1[idx]
                his5 = flat_his5[idx]
                his2 = flat_his2[idx]
                his3 = flat_his3[idx]

                c1 = get_cmax_at_t(his1, t)
                c5 = get_cmax_at_t(his5, t)
                c2 = get_cmax_at_t(his2, t)
                c3 = get_cmax_at_t(his3, t)

                cmax_c1_t.append(c1)
                cmax_c5_t.append(c5)
                cmax_c2_t.append(c2)
                cmax_c3_t.append(c3)

                final_c1 = get_final_cmax_under_budget(his1, t)
                final_c5 = get_final_cmax_under_budget(his5, t)
                final_c2 = get_final_cmax_under_budget(his2, t)
                final_c3 = get_final_cmax_under_budget(his3, t)
                final_c1_at_t_runs.append(final_c1)
                final_c5_at_t_runs.append(final_c5)
                final_c2_at_t_runs.append(final_c2)
                final_c3_at_t_runs.append(final_c3)

                cmax_gap5_runs.append((c1 - c5) / c1 if c1 > 0 else 0.0)
                cmax_gap2_runs.append((c1 - c2) / c1 if c1 > 0 else 0.0)
                cmax_gap3_runs.append((c1 - c3) / c1 if c1 > 0 else 0.0)

                t_c1_self = get_time_to_target_with_budget(his1, final_c1, t)
                t_c5_self = get_time_to_target_with_budget(his5, final_c5, t)
                t_c2_self = get_time_to_target_with_budget(his2, final_c2, t)
                t_c3_self = get_time_to_target_with_budget(his3, final_c3, t)
                t_c5_final = get_time_to_target_with_budget(his5, final_c1, t)
                t_c2_final = get_time_to_target_with_budget(his2, final_c1, t)
                t_c3_final = get_time_to_target_with_budget(his3, final_c1, t)

                time_c1_self_reach_runs.append(t_c1_self)
                time_c5_self_reach_runs.append(t_c5_self)
                time_c2_self_reach_runs.append(t_c2_self)
                time_c3_self_reach_runs.append(t_c3_self)
                time_c5_reach_c1_runs.append(t_c5_final)
                time_c2_reach_c1_runs.append(t_c2_final)
                time_c3_reach_c1_runs.append(t_c3_final)

                t_c1_safe = max(t_c1_self, MIN_TIME_RES)
                t_c5_safe = max(t_c5_final, MIN_TIME_RES)
                t_c2_safe = max(t_c2_final, MIN_TIME_RES)
                t_c3_safe = max(t_c3_final, MIN_TIME_RES)

                speedup_mixed_runs.append(t_c1_safe / t_c5_safe)
                speedup_mixed2_runs.append(t_c1_safe / t_c2_safe)
                speedup_mixed3_runs.append(t_c1_safe / t_c3_safe)
                saving_mixed_runs.append((t_c1_safe - t_c5_safe) / t_c1_safe)
                saving_mixed2_runs.append((t_c1_safe - t_c2_safe) / t_c1_safe)
                saving_mixed3_runs.append((t_c1_safe - t_c3_safe) / t_c1_safe)

            mean_c1 = safe_mean(cmax_c1_t)
            std_c1 = safe_std(cmax_c1_t)
            mean_c5 = safe_mean(cmax_c5_t)
            std_c5 = safe_std(cmax_c5_t)
            mean_c2 = safe_mean(cmax_c2_t)
            std_c2 = safe_std(cmax_c2_t)
            mean_c3 = safe_mean(cmax_c3_t)
            std_c3 = safe_std(cmax_c3_t)

            final_mean_c1_at_t = safe_mean(final_c1_at_t_runs)
            final_mean_c5_at_t = safe_mean(final_c5_at_t_runs)
            final_mean_c2_at_t = safe_mean(final_c2_at_t_runs)
            final_mean_c3_at_t = safe_mean(final_c3_at_t_runs)

            c_gap5 = safe_mean(cmax_gap5_runs)
            c_gap2 = safe_mean(cmax_gap2_runs)
            c_gap3 = safe_mean(cmax_gap3_runs)
            speedup_mixed_at_t = safe_mean(speedup_mixed_runs)
            speedup_mixed2_at_t = safe_mean(speedup_mixed2_runs)
            speedup_mixed3_at_t = safe_mean(speedup_mixed3_runs)
            time_saving_ratio_t_mixed = safe_mean(saving_mixed_runs)
            time_saving_ratio_t_mixed2 = safe_mean(saving_mixed2_runs)
            time_saving_ratio_t_mixed3 = safe_mean(saving_mixed3_runs)

            mean_time_c1_to_final_c1_at_t = safe_mean(time_c1_self_reach_runs)
            mean_time_c5_to_final_c1_at_t = safe_mean(time_c5_reach_c1_runs)
            mean_time_c2_to_final_c1_at_t = safe_mean(time_c2_reach_c1_runs)
            mean_time_c3_to_final_c1_at_t = safe_mean(time_c3_reach_c1_runs)
            mean_time_c5_to_final_c5_at_t = safe_mean(time_c5_self_reach_runs)
            mean_time_c2_to_final_c2_at_t = safe_mean(time_c2_self_reach_runs)
            mean_time_c3_to_final_c3_at_t = safe_mean(time_c3_self_reach_runs)

            p_m5_two = wilcoxon_pvalue(cmax_c5_t, cmax_c1_t, alternative="two-sided")
            p_m5_less = wilcoxon_pvalue(cmax_c5_t, cmax_c1_t, alternative="less")
            p_m2_two = wilcoxon_pvalue(cmax_c2_t, cmax_c1_t, alternative="two-sided")
            p_m2_less = wilcoxon_pvalue(cmax_c2_t, cmax_c1_t, alternative="less")
            p_m3_two = wilcoxon_pvalue(cmax_c3_t, cmax_c1_t, alternative="two-sided")
            p_m3_less = wilcoxon_pvalue(cmax_c3_t, cmax_c1_t, alternative="less")

            summary_data.append({
                "Num_Instances": num_instances,
                "Repeats_Per_Instance": num_repeats,
                "Total_Runs": total_runs,
                "Iteration_Stop": stop_val,
                "Alpha_t": current_alpha,
                "Time(s)": t,
                "Cmax_CCEA(No GNN)": mean_c1,
                "Std_CCEA": std_c1,
                "Cmax_GNN(both)": mean_c5,
                "Std_GNN(both)": std_c5,
                "Cmax_Gap(both)": c_gap5,
                "p_two_sided(both_vs_NoGNN)": p_m5_two,
                "p_one_sided_less(both_vs_NoGNN)": p_m5_less,
                "Cmax_GNN(worker only)": mean_c2,
                "Std_GNN(worker only)": std_c2,
                "Cmax_Gap(worker only)": c_gap2,
                "p_two_sided(worker only_vs_NoGNN)": p_m2_two,
                "p_one_sided_less(worker only_vs_NoGNN)": p_m2_less,
                "Cmax_GNN(batch only)": mean_c3,
                "Std_GNN(batch only)": std_c3,
                "Cmax_Gap(batch only)": c_gap3,
                "p_two_sided(batch only_vs_NoGNN)": p_m3_two,
                "p_one_sided_less(batch only_vs_NoGNN)": p_m3_less,
                "Time_Saving_Ratio_at_t(both)": time_saving_ratio_t_mixed,
                "Time_Saving_Ratio_at_t(worker only)": time_saving_ratio_t_mixed2,
                "Time_Saving_Ratio_at_t(batch only)": time_saving_ratio_t_mixed3,
                "Mean_Final_Cmax_CCEA(No GNN)_at_t": final_mean_c1_at_t,
                "Mean_Final_Cmax_GNN(both)_at_t": final_mean_c5_at_t,
                "Mean_Final_Cmax_GNN(worker only)_at_t": final_mean_c2_at_t,
                "Mean_Final_Cmax_GNN(batch only)_at_t": final_mean_c3_at_t,
                "Time_NoGNN_to_Final_Cmax_of_NoGNN_at_t": mean_time_c1_to_final_c1_at_t,
                "Time_both_to_Final_Cmax_of_both_at_t": mean_time_c5_to_final_c5_at_t,
                "Time_worker only_to_Final_Cmax_of_worker only_at_t": mean_time_c2_to_final_c2_at_t,
                "Time_batch only_to_Final_Cmax_of_batch only_at_t": mean_time_c3_to_final_c3_at_t,
                "Time_both_to_Final_Cmax_of_NoGNN_at_t": mean_time_c5_to_final_c1_at_t,
                "Time_worker only_to_Final_Cmax_of_NoGNN_at_t": mean_time_c2_to_final_c1_at_t,
                "Time_batch only_to_Final_Cmax_of_NoGNN_at_t": mean_time_c3_to_final_c1_at_t,
                "Speedup_both_vs_NoGNN_at_t": speedup_mixed_at_t,
                "Speedup_worker only_vs_NoGNN_at_t": speedup_mixed2_at_t,
                "Speedup_batch only_vs_NoGNN_at_t": speedup_mixed3_at_t,
            })

    path = f"outputsCon260408ablationV3/{instance_seeds}"
    os.makedirs(path, exist_ok=True)
    df_summary = pd.DataFrame(summary_data)
    df_table1 = build_ablation_table1(histories, iteration_stops, W, J, instance_seeds, num_repeats)
    df_table2 = build_ablation_table2(histories, iteration_stops, stop_time, W, J, instance_seeds, num_repeats)
    
    df_table1.to_csv(os.path.join(path, f"W{W}_J{J}_Paper_Table1_FinalPerformance_5instances.csv"), index=False, encoding="utf-8-sig")
    df_table2.to_csv(os.path.join(path, f"W{W}_J{J}_Paper_Table2_AccelerationStats_5instances.csv"), index=False, encoding="utf-8-sig")

    fig1_dfs = {}

    for i in range(num_instances):
        representative_instance_id = i
        representative_instance_seed = instance_seeds[representative_instance_id]

        rep_histories_for_fig = {
            'CCEA1': histories[paper_fig_stop]['CCEA1'][representative_instance_id],
            'CCEA2': histories[paper_fig_stop]['CCEA2'][representative_instance_id],
            'CCEA3': histories[paper_fig_stop]['CCEA3'][representative_instance_id],
            'CCEA5': histories[paper_fig_stop]['CCEA5'][representative_instance_id],
        }

        df_fig1 = build_paper_figure_data(rep_histories_for_fig, stop_time)
        df_fig1.insert(0, "Representative_Instance_Seed", representative_instance_seed)

        fig1_dfs[f"Fig1_Instance_{i+1}"] = df_fig1

        csv_path = os.path.join(
            path,
            f"W{W}_J{J}_Paper_Fig1_Data_Stop{paper_fig_stop}_InstanceSeed{representative_instance_seed}.csv"
        )
        df_fig1.to_csv(csv_path, index=False, encoding="utf-8-sig")

        fig_path = os.path.join(
            path,
            f"W{W}_J{J}_Paper_Fig1_Stop{paper_fig_stop}_Seed{representative_instance_seed}_Convergence.png"
        )
        plot_paper_ablation_figure(
            df_fig1,
            fig_path,
            W,
            J,
            paper_fig_stop,
            representative_instance_seed,
        )

    per_sec_dfs = {}
    trajectory_stops_to_export = [20, 50]
    time_grid = build_mixed_time_grid(max_record_sec)
    time_cols = [
        f"{t:.1f}s" if abs(t - round(t)) > 1e-9 else f"{int(round(t))}s"
        for t in time_grid
    ]

    for stop_val in trajectory_stops_to_export:
        for algo in ['CCEA1', 'CCEA2', 'CCEA5', 'CCEA3']:
            sec_data = []
            for inst_id, inst_seed in enumerate(instance_seeds):
                for rep_id in range(num_repeats):
                    full_history = histories[stop_val][algo][inst_id][rep_id]
                    traj = get_trajectory_by_time_grid(full_history, time_grid)
                    sec_data.append([
                        inst_id + 1,
                        inst_seed,
                        f"Inst{inst_id + 1}_Run{rep_id + 1}",
                    ] + traj)

            cols = ["Instance_ID", "Instance_Seed", "Run_ID"] + time_cols
            df_sec = pd.DataFrame(sec_data, columns=cols)
            mean_values = df_sec.iloc[:, 3:].mean().tolist()
            std_values = df_sec.iloc[:, 3:].std().tolist()
            df_sec.loc[len(df_sec)] = ["ALL", "ALL", "Mean"] + mean_values
            df_sec.loc[len(df_sec)] = ["ALL", "ALL", "Std"] + std_values
            per_sec_dfs[f"Traj_Stop_{stop_val}_{algo}"] = df_sec

    excel_filename = os.path.join(path, f"W{W}_J{J}_GNN_ablation_Result_5instances.xlsx")
    with pd.ExcelWriter(excel_filename) as writer:
        pd.DataFrame(instance_infos).to_excel(writer, sheet_name="Instance_Info", index=False)
        df_table1.to_excel(writer, sheet_name="Paper_Table1", index=False)
        df_table2.to_excel(writer, sheet_name="Paper_Table2", index=False)
        for sheet_name, df_fig in fig1_dfs.items():
            df_fig.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        for stop_val in iteration_stops:
            df_sub = df_summary[df_summary['Iteration_Stop'] == stop_val]
            df_sub.to_excel(writer, sheet_name=f"Summary_Stop_{stop_val}", index=False)
        for sheet_name, df_sec in per_sec_dfs.items():
            safe_sheet_name = sheet_name[:31]
            df_sec.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    # paper_fig_path = os.path.join(
    #     path,
    #     f"W{W}_J{J}_Paper_Fig1_Stop{paper_fig_stop}_Seed{representative_instance_seed}_Convergence.png"
    # )
    # plot_paper_ablation_figure(
    #     df_fig1,
    #     paper_fig_path,
    #     W,
    #     J,
    #     paper_fig_stop,
    #     representative_instance_seed,
    # )

    print(f"\n数据处理完毕！已生成结果文件：{excel_filename}")
    print(f"论文表1已保存为: {os.path.join(path, f'W{W}_J{J}_Paper_Table1_FinalPerformance_5instances.csv')}")
    print(f"论文表2已保存为: {os.path.join(path, f'W{W}_J{J}_Paper_Table2_AccelerationStats_5instances.csv')}")
    print(f"每个实例的论文图1数据与收敛图已保存到目录: {path}")

    for stop_val in iteration_stops:
        plot_multi_xy_groups(
            {
                "GNN(both)": flatten_runs(histories[stop_val]['CCEA5']),
                "NO_GNN(CCEA)": flatten_runs(histories[stop_val]['CCEA1']),
                "GNN(worker only)": flatten_runs(histories[stop_val]['CCEA2']),
                "GNN(batch only)": flatten_runs(histories[stop_val]['CCEA3']),
            },
            save_dir=path,
            prefix=f"W{W}_J{J}_Stop{stop_val}_compare_5instances",
            ci="stderr",
            grid="union",
            max_runs_plot=total_runs,
            truncate="min_time",
        )


if __name__ == "__main__":
    # test_build_and_plot_paper_figure()
    W = [25]
    J = [200]
    for w in W:
        for j in J:
            seed = 42
            data_generate(num_repeats=20, W=w, J=j, seed=seed, paper_fig_stop=50, num_instances=5)
