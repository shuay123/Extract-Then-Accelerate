# -*- coding: utf-8 -*-
"""
ccea_rk_branch.py
可运行分支版本：Random Keys（优先级染色体）+ 图着色/贪心解码器 + 动态调度冲突更新 + gamma 自适应软约束

与“最小改动版”不同：
- 完整替换染色体与 GA 算子（不再使用 OX / swap_mutation）；
- 每个个体是连续 keys（Random Keys），通过解码器生成 formation/schedule；
- 冲突矩阵在解码阶段通过 Δcost 直接引导分配决策；
- 调度冲突权重 w_eff 由当前 best_formation 动态更新；
- 仍复用你项目的 CalculateFitness 作为真实 makespan 评估器。

放置位置（建议）：
- 保存为 metaheuristics/ccea/ccea_rk_branch.py
- 同目录放置：conflict_dynamic.py / rk_decoders.py / rk_operators.py
- 在 main 或入口脚本中 import 并运行 main()

配置建议：
- config_seru.worker_confidence_path / batch_confidence_path 指向 0~1 冲突矩阵（.npy/.csv/.xlsx）
- config_seru.num_of_serus（建议显式给定）
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
# 添加项目根目录到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import numpy as np
import pandas as pd

from utils.excel_utils import ExcelDataLoader
from utils.config_loader import ConfigLoader

from problem.pure_seru.pure_seru_entities import Seru, SeruFormation, SeruSchedule, Solution
from problem.pure_seru.calculate_fitness import CalculateFitness

from metaheuristics.ccea.conflict_dynamic import ConflictModel
from metaheuristics.ccea.rk_decoders import decode_formation, decode_schedule
from metaheuristics.ccea.rk_operators import (
    blend_crossover, uniform_crossover,
    gaussian_mutation, reset_mutation,
    tournament_select
)


def load_conf_matrix(path: Optional[str], size: int, one_based: bool = True) -> np.ndarray:
    """
    支持 .npy / .csv / .xlsx（可在 path 里用 file.xlsx::Sheet1 指定 sheet）
    """
    n = size + 1 if one_based else size
    if not path:
        return np.zeros((n, n), dtype=float)

    if path.endswith(".npy"):
        M = np.load(path)
    elif path.endswith(".csv"):
        M = np.loadtxt(path, delimiter=",")
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        if "::" in path:
            file_path, sheet = path.split("::", 1)
            df = pd.read_excel(file_path, sheet_name=sheet, header=None)
        else:
            df = pd.read_excel(path, sheet_name=0, header=None)
        M = df.values
    else:
        raise ValueError(f"Unsupported matrix format: {path}")

    M = np.array(M, dtype=float)
    M = np.clip(M, 0.0, 1.0)
    if M.shape[0] != n or M.shape[1] != n:
        M2 = np.zeros((n, n), dtype=float)
        r = min(n, M.shape[0]); c = min(n, M.shape[1])
        M2[:r, :c] = M[:r, :c]
        M = M2
    return M


@dataclass
class FormationInd:
    keys: np.ndarray
    fitness: float = float("inf")
    makespan: float = float("inf")
    r_base: float = 0.0


@dataclass
class ScheduleInd:
    keys_ins: np.ndarray
    keys_seq: np.ndarray
    fitness: float = float("inf")
    makespan: float = float("inf")
    r_base: float = 0.0


@dataclass
class Elite:
    fitness: float
    makespan: float
    formation: SeruFormation
    schedule: SeruSchedule


class CCEA_RK:
    def __init__(self):
        self.config_seru = ConfigLoader.get_config("config_seru")
        self.config_ccea = ConfigLoader.get_config("config_ccea")

        # 读数据
        self.loader = ExcelDataLoader()
        self.loader.read_data(excel_path=self.config_seru.seru_data_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.due_dates_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.batch_types_path, config_sheet=self.config_seru)

        # singleton loader（兼容你的 CalculateFitness 用法）
        self.excel = ExcelDataLoader.instance() if hasattr(ExcelDataLoader, "instance") else self.loader

        # 尺寸
        self.W = int(getattr(self.config_seru, "num_of_workers", 0) or 0)
        self.B = int(getattr(self.config_seru, "num_of_batches", 0) or 0)

        # Seru 数量（建议显式配置；否则用 sqrt(W) 兜底）
        self.K = int(
            getattr(self.config_seru, "num_of_serus", 0)
            or getattr(self.config_seru, "num_of_seru", 0)
            or getattr(self.config_ccea, "num_of_serus", 0)
            or max(1, int(round(math.sqrt(max(self.W, 1)))))
        )

        # 冲突矩阵路径
        worker_path = getattr(self.config_seru, "worker_confidence_path", None) \
                      or getattr(self.config_seru, "worker_conf_path", None) \
                      or getattr(self.config_ccea, "worker_confidence_path", None)
        batch_path = getattr(self.config_seru, "batch_confidence_path", None) \
                     or getattr(self.config_seru, "batch_conf_path", None) \
                     or getattr(self.config_ccea, "batch_confidence_path", None)

        # 超参
        rho = float(getattr(self.config_ccea, "conflict_rho", 1.5))
        eps = float(getattr(self.config_ccea, "conflict_eps", 1e-6))
        alpha = float(getattr(self.config_ccea, "dynamic_conflict_alpha", 1.0))

        thw = getattr(self.config_ccea, "theta_hard_worker", None)
        thb = getattr(self.config_ccea, "theta_hard_batch", None)
        thw = float(thw) if thw is not None else None
        thb = float(thb) if thb is not None else None

        worker_conf = load_conf_matrix(worker_path, size=self.W, one_based=True)
        batch_conf = load_conf_matrix(batch_path, size=self.B, one_based=True)

        self.conflict = ConflictModel(
            worker_conf=worker_conf,
            batch_conf=batch_conf,
            rho=rho,
            eps=eps,
            alpha=alpha,
            theta_hard_worker=thw,
            theta_hard_batch=thb,
        )

        # 软约束控制：r* = 10%
        self.r_target = float(getattr(self.config_ccea, "conflict_r_target", 0.10))
        self.eta = float(getattr(self.config_ccea, "conflict_eta", 0.1))
        self.gamma = float(getattr(self.config_ccea, "conflict_gamma_init", 1.0))
        self.gamma_min = float(getattr(self.config_ccea, "conflict_gamma_min", 0.0))
        self.gamma_max = float(getattr(self.config_ccea, "conflict_gamma_max", 5.0))

        # GA 参数
        self.pop = int(getattr(self.config_ccea, "population_size", 50))
        self.tourn_k = int(getattr(self.config_ccea, "tournament_k", 2))
        self.p_cross = float(getattr(self.config_ccea, "p_cross", 0.9))
        self.p_uniform = float(getattr(self.config_ccea, "p_uniform_cross", 0.2))
        self.mut_sigma = float(getattr(self.config_ccea, "rk_mut_sigma", 0.15))
        self.mut_p = float(getattr(self.config_ccea, "rk_mut_p", 0.15))
        self.reset_p = float(getattr(self.config_ccea, "rk_reset_p", 0.01))

        self.ls_steps = int(getattr(self.config_ccea, "rk_local_search_steps", 15))
        self.max_runtime = float(getattr(self.config_ccea, "max_runtime", 10.0))

        seed = int(getattr(self.config_ccea, "seed", 42))
        self.rng = np.random.default_rng(seed)

        # 双精英池（保底）
        self.elite_fit: Optional[Elite] = None
        self.elite_ms: Optional[Elite] = None

    # -------------------------
    # gamma 自适应
    # -------------------------
    def _update_gamma(self, r_avg: float):
        g = self.gamma * math.exp(self.eta * (float(r_avg) - self.r_target))
        self.gamma = max(self.gamma_min, min(self.gamma_max, g))

    # -------------------------
    # 解码：keys -> formation/schedule
    # -------------------------
    def _decode_formation_to_entity(self, keysW: np.ndarray, partner_schedule: SeruSchedule, M_ref: float) -> Tuple[SeruFormation, List[List[int]]]:
        seru_workers = decode_formation(
            keys_workers=keysW,
            num_serus=self.K,
            conflict=self.conflict,
            worker_to_product=getattr(self.excel, "worker_to_product_dict", {}) or {},
            worker_to_task=getattr(self.excel, "worker_to_task_dict", {}) or {},
            num_workers=self.W,
            max_multiple_task=float(getattr(self.config_seru, "max_num_of_multiple_task", 0) or 0),
            gamma=self.gamma,
            M_ref=M_ref,
        )
        formation = SeruFormation()
        formation.seru_set = [Seru(workers_set=list(ws)) for ws in seru_workers]
        return formation, seru_workers

    def _decode_schedule_to_entity(self, keys_ins: np.ndarray, keys_seq: np.ndarray, seru_workers: List[List[int]], M_ref: float, use_dynamic: bool) -> SeruSchedule:
        batches_assignment = decode_schedule(
            keys_ins=keys_ins,
            keys_seq=keys_seq,
            seru_workers=seru_workers,
            conflict=self.conflict,
            batch_to_product=getattr(self.excel, "batch_to_product_dict", {}) or {},
            worker_to_product=getattr(self.excel, "worker_to_product_dict", {}) or {},
            worker_to_task=getattr(self.excel, "worker_to_task_dict", {}) or {},
            num_workers=self.W,
            max_multiple_task=float(getattr(self.config_seru, "max_num_of_multiple_task", 0) or 0),
            gamma=self.gamma,
            M_ref=M_ref,
            use_dynamic=use_dynamic,
        )
        schedule = SeruSchedule()
        schedule.batches_assignment = batches_assignment
        return schedule

    # -------------------------
    # 评估：CalculateFitness（真实） + r_base（控制 gamma）
    # -------------------------
    def _evaluate(self, formation: SeruFormation, schedule: SeruSchedule, M_ref: float) -> Tuple[float, float, float]:
        # CalculateFitness 会在内部 init
        sol = Solution(formation=formation, schedule=schedule)
        CalculateFitness.calculate_fitness(sol, self.config_seru)
        M = float(getattr(sol, "makespan", 0.0) or getattr(sol, "fitness", 0.0) or 0.0)

        # worker_seru 映射
        worker_seru: Dict[int, int] = {}
        for s_idx, seru in enumerate(getattr(formation, "seru_set", []) or []):
            for wid in getattr(seru, "workers_set", []) or []:
                worker_seru[int(wid)] = int(s_idx)

        # batch_seru 映射（基于 batches_assignment 规则：第 s 组映射到 seru s）
        batch_seru: Dict[int, int] = {}
        ba = getattr(schedule, "batches_assignment", []) or []
        for s_idx, group in enumerate(ba):
            for bid in group:
                batch_seru[int(bid)] = int(s_idx)

        PW = self.conflict.worker_violation_base(worker_seru)
        PB = self.conflict.batch_violation_base(batch_seru)
        r = float(PW + PB) / float(self.conflict.W_tot)
        fit = M + float(self.gamma) * float(M_ref) * float(r)
        return fit, M, r

    # -------------------------
    # 局部搜索（keys 微扰）
    # -------------------------
    def _local_search_keys(self, keys: np.ndarray) -> np.ndarray:
        # 最小实现：随机交换两位（在 keys 空间等价于改变优先级）
        y = keys.copy()
        if len(y) >= 2:
            i, j = self.rng.choice(len(y), size=2, replace=False)
            y[i], y[j] = y[j], y[i]
        return y

    # -------------------------
    # 精英池
    # -------------------------
    def _update_elites(self, fit: float, M: float, formation: SeruFormation, schedule: SeruSchedule):
        if self.elite_fit is None or fit < self.elite_fit.fitness:
            self.elite_fit = Elite(fitness=fit, makespan=M, formation=formation, schedule=schedule)
        if self.elite_ms is None or M < self.elite_ms.makespan:
            self.elite_ms = Elite(fitness=fit, makespan=M, formation=formation, schedule=schedule)

    # -------------------------
    # 主流程
    # -------------------------
    def run(self) -> Tuple[SeruFormation, SeruSchedule, Solution]:
        # 初始化种群
        PSF = [FormationInd(keys=self.rng.random(self.W)) for _ in range(self.pop)]
        PSS = [ScheduleInd(keys_ins=self.rng.random(self.B), keys_seq=self.rng.random(self.B)) for _ in range(self.pop)]

        # 初始化 best（用随机第一个）
        M_ref = 1.0
        # 先随便解码一个 schedule（不动态），再解码 formation
        tmp_seru_workers = [[w] for w in range(1, self.K + 1)] + [[] for _ in range(max(0, self.K - self.W))]
        best_schedule = self._decode_schedule_to_entity(PSS[0].keys_ins, PSS[0].keys_seq, tmp_seru_workers[:self.K], M_ref=M_ref, use_dynamic=False)
        best_formation, best_seru_workers = self._decode_formation_to_entity(PSF[0].keys, best_schedule, M_ref=M_ref)

        # 用 best_formation 更新动态 profile（供后续调度使用）
        # schedule 解码函数内部也会 update_seru_profiles，但这里提前做一次更清晰
        # ——这里用 decode_schedule 内部 cap 计算更一致，因此不额外计算

        # 先评估一把，得到合理 M_ref
        fit0, M0, r0 = self._evaluate(best_formation, best_schedule, M_ref=max(1.0, M_ref))
        M_ref = max(1.0, M0)

        self._update_elites(fit0, M0, best_formation, best_schedule)
        best_solution = Solution(formation=best_formation, schedule=best_schedule)
        best_solution.fitness = fit0
        best_solution.makespan = M0

        start = time.time()
        it = 0
        while time.time() - start < self.max_runtime:
            it += 1
            M_ref = max(1.0, float(getattr(best_solution, "makespan", 1.0) or 1.0))

            # -------------------------
            # A) 进化 PSF（固定 best_schedule）
            # -------------------------
            r_list = []
            fvals = np.zeros(self.pop, dtype=float)
            for idx, ind in enumerate(PSF):
                formation, seru_workers = self._decode_formation_to_entity(ind.keys, best_schedule, M_ref=M_ref)
                fit, M, r = self._evaluate(formation, best_schedule, M_ref=M_ref)
                ind.fitness, ind.makespan, ind.r_base = fit, M, r
                fvals[idx] = fit
                r_list.append(r)
                self._update_elites(fit, M, formation, best_schedule)

            self._update_gamma(sum(r_list) / max(1, len(r_list)))

            # 产生下一代 PSF
            new_PSF: List[FormationInd] = []
            while len(new_PSF) < self.pop:
                i1 = tournament_select(fvals, self.tourn_k, self.rng)
                i2 = tournament_select(fvals, self.tourn_k, self.rng)
                p1, p2 = PSF[i1].keys, PSF[i2].keys
                if self.rng.random() < self.p_cross:
                    if self.rng.random() < self.p_uniform:
                        c1, c2 = uniform_crossover(p1, p2, self.rng)
                    else:
                        c1, c2 = blend_crossover(p1, p2, self.rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = gaussian_mutation(c1, sigma=self.mut_sigma, p=self.mut_p, rng=self.rng)
                c2 = gaussian_mutation(c2, sigma=self.mut_sigma, p=self.mut_p, rng=self.rng)
                c1 = reset_mutation(c1, p=self.reset_p, rng=self.rng)
                c2 = reset_mutation(c2, p=self.reset_p, rng=self.rng)
                new_PSF.append(FormationInd(keys=c1))
                if len(new_PSF) < self.pop:
                    new_PSF.append(FormationInd(keys=c2))
            PSF = new_PSF

            # 更新 best_formation（取本代 PSF 的最优个体解码结果）
            # ——最稳做法：直接从 elite_fit / elite_ms 获取；这里用 elite_fit 对齐“综合目标”
            best_formation = self.elite_fit.formation if self.elite_fit is not None else best_formation
            best_seru_workers = [list(getattr(s, "workers_set", []) or []) for s in getattr(best_formation, "seru_set", [])]

            # -------------------------
            # B) 进化 PSS（固定 best_formation + 动态 w_eff）
            # -------------------------
            r_list = []
            svals = np.zeros(self.pop, dtype=float)
            for idx, ind in enumerate(PSS):
                schedule = self._decode_schedule_to_entity(ind.keys_ins, ind.keys_seq, best_seru_workers, M_ref=M_ref, use_dynamic=True)
                fit, M, r = self._evaluate(best_formation, schedule, M_ref=M_ref)
                ind.fitness, ind.makespan, ind.r_base = fit, M, r
                svals[idx] = fit
                r_list.append(r)
                self._update_elites(fit, M, best_formation, schedule)

            self._update_gamma(sum(r_list) / max(1, len(r_list)))

            # 产生下一代 PSS
            new_PSS: List[ScheduleInd] = []
            while len(new_PSS) < self.pop:
                i1 = tournament_select(svals, self.tourn_k, self.rng)
                i2 = tournament_select(svals, self.tourn_k, self.rng)
                p1_ins, p2_ins = PSS[i1].keys_ins, PSS[i2].keys_ins
                p1_seq, p2_seq = PSS[i1].keys_seq, PSS[i2].keys_seq

                if self.rng.random() < self.p_cross:
                    if self.rng.random() < self.p_uniform:
                        c1_ins, c2_ins = uniform_crossover(p1_ins, p2_ins, self.rng)
                        c1_seq, c2_seq = uniform_crossover(p1_seq, p2_seq, self.rng)
                    else:
                        c1_ins, c2_ins = blend_crossover(p1_ins, p2_ins, self.rng)
                        c1_seq, c2_seq = blend_crossover(p1_seq, p2_seq, self.rng)
                else:
                    c1_ins, c2_ins = p1_ins.copy(), p2_ins.copy()
                    c1_seq, c2_seq = p1_seq.copy(), p2_seq.copy()

                c1_ins = gaussian_mutation(c1_ins, sigma=self.mut_sigma, p=self.mut_p, rng=self.rng)
                c2_ins = gaussian_mutation(c2_ins, sigma=self.mut_sigma, p=self.mut_p, rng=self.rng)
                c1_seq = gaussian_mutation(c1_seq, sigma=self.mut_sigma, p=self.mut_p, rng=self.rng)
                c2_seq = gaussian_mutation(c2_seq, sigma=self.mut_sigma, p=self.mut_p, rng=self.rng)

                c1_ins = reset_mutation(c1_ins, p=self.reset_p, rng=self.rng)
                c2_ins = reset_mutation(c2_ins, p=self.reset_p, rng=self.rng)
                c1_seq = reset_mutation(c1_seq, p=self.reset_p, rng=self.rng)
                c2_seq = reset_mutation(c2_seq, p=self.reset_p, rng=self.rng)

                new_PSS.append(ScheduleInd(keys_ins=c1_ins, keys_seq=c1_seq))
                if len(new_PSS) < self.pop:
                    new_PSS.append(ScheduleInd(keys_ins=c2_ins, keys_seq=c2_seq))
            PSS = new_PSS

            # 更新 best_schedule（同样使用 elite_fit）
            best_schedule = self.elite_fit.schedule if self.elite_fit is not None else best_schedule

            # -------------------------
            # C) 更新 best_solution（以 elite_fit 为主；并保留 elite_ms）
            # -------------------------
            if self.elite_fit is not None:
                best_solution = Solution(formation=self.elite_fit.formation, schedule=self.elite_fit.schedule)
                best_solution.fitness = self.elite_fit.fitness
                best_solution.makespan = self.elite_fit.makespan

            if it % 1 == 0 and self.elite_fit is not None and self.elite_ms is not None:
                elapsed = time.time() - start
                print(
                    f"Iter {it} | fit={self.elite_fit.fitness:.6f} ms={self.elite_fit.makespan:.6f} "
                    f"| elite_ms={self.elite_ms.makespan:.6f} | r*={self.r_target:.2f} gamma={self.gamma:.4f} "
                    f"| t={elapsed:.2f}s"
                )

        # 返回综合最优（elite_fit）；如果你只要最短工期，可改返回 elite_ms
        final = self.elite_fit if self.elite_fit is not None else self.elite_ms
        if final is None:
            final = Elite(fitness=best_solution.fitness, makespan=best_solution.makespan,
                          formation=best_solution.formation, schedule=best_solution.schedule)

        sol = Solution(formation=final.formation, schedule=final.schedule)
        sol.fitness = final.fitness
        sol.makespan = final.makespan
        return final.formation, final.schedule, sol


def main():
    ConfigLoader.preload_all()
    ccea = CCEA_RK()
    best_formation, best_schedule, best_solution = ccea.run()

    print("---- Final (elite_fit) ----")
    print("makespan:", getattr(best_solution, "makespan", None))
    print("fitness :", getattr(best_solution, "fitness", None))

    if hasattr(best_formation, "seru_set"):
        for i, seru in enumerate(best_formation.seru_set, start=1):
            ws = getattr(seru, "workers_set", []) or []
            print(f"Seru {i}: {len(ws)} workers")

    # schedule 输出：每个 seru 对应的批次列表
    ba = getattr(best_schedule, "batches_assignment", []) or []
    for i, g in enumerate(ba, start=1):
        print(f"Seru {i} batches:", g)


if __name__ == "__main__":
    main()
