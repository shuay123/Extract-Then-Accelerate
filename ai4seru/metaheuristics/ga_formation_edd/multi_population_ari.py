# island_ga_simple.py
import random, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from utils.config_loader import ConfigLoader
from problem.pure_seru.pure_seru_entities import SeruFormation
from metaheuristics.ga_formation_edd.ga_formation import GA
from utils.cal_adjusted_rand_score import calculate_ari


def _run_single_island(seed: int) -> Tuple[float, List[SeruFormation]]:
    """子进程：跑一份 GA 并返回 (best_fitness, archive)"""
    ConfigLoader.preload_all()  # 子进程自己加载配置
    random.seed(seed)
    ga = GA()
    best_f, _, best_sol, arch = ga.run()
    return best_sol.fitness, arch


def run_multi_population_ari()-> (List[SeruFormation]):
    ConfigLoader.preload_all()
    config_ga = ConfigLoader.get_config("config_ga")
    config_drl = ConfigLoader.get_config("config_drl").config_data
    n_islands = config_ga.n_islands  # 岛屿数
    seeds = [int(time.time() * 1e6) + i for i in range(n_islands)]

    all_archives: List[SeruFormation] = []  # 全局归并
    with ProcessPoolExecutor(max_workers=n_islands) as pool:
        futures = [pool.submit(_run_single_island, s) for s in seeds]
        for idx, fut in enumerate(as_completed(futures), 1):
            best_fit, arch = fut.result()
            print(f"岛 {idx} best = {best_fit:.4f},  archive size = {len(arch)}")
            all_archives.extend(arch)  # 直接拼接

    # 这里 all_archives 就是所有岛屿的 archive 总和
    print(f"\n合并后 archive 总大小: {len(all_archives)}")
    # ---------- 计算 ARI 均值并选出差异最大的 N 个 ----------
    ari_scores = calculate_ari(all_archives)  # List[float]，一一对应
    # 把 (formation, ari) 组合后按 ari 升序
    formation_by_diversity = sorted(zip(all_archives, ari_scores), key=lambda x: x[1])

    n = config_drl['train']['num_envs']  # 目标数量
    selected, seen = [], set()  # seen 用于“同分只取一个”

    for fm, score in formation_by_diversity:
        key = round(score, 8)  # ↓ 浮点数用 round 或 math.isclose 规避 0.2000001 ≠ 0.2
        if key in seen:  # 已有同样 ARI ➜ 跳过
            continue
        selected.append((fm, score))
        seen.add(key)
        if len(selected) == n:  # 收够 n 个就停
            break

    diverse_formations = [fm for fm, _ in selected]

    print(f"为 DRL 选取的 {n} 个差异最大 Seru 构造（ARI 最低，且同分去重）：")
    for i, (fm, score) in enumerate(selected, 1):
        print(f"{i:>2}. fitness = {fm.fitness:.4f},  mean‑ARI = {score:.4f}")

    return diverse_formations


if __name__ == "__main__":
    run_multi_population_ari()
