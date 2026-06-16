# -*- coding: utf-8 -*-
"""
最小改动版：在现有 CCEA（PSF+PSS）框架中引入“冲突置信度矩阵（0~1）”的软约束，并实现：
1) 适应值 = makespan + gamma * M_ref * r（r 为归一化违背率，目标约 10%）
2) gamma 自适应更新（每轮/每阶段按平均违背率更新）
3) 调度阶段的冲突权重随当前 best_formation 动态更新（条件化 w_eff(s,p,q)）
4) 不改动现有染色体（formation_code/schedule_code）与 GA 算子（order_crossover/swap_mutation），只改评估与选择/局部搜索逻辑

使用方式（建议）：
- 将本文件替换 metaheuristics/ccea/ccea.py（或重命名后在 main 中 import）
- 在 config_seru 或 config_ccea 中提供 worker_confidence_path / batch_confidence_path（支持 .npy/.csv/.xlsx）
- 如果不提供矩阵路径，会退化为无先验（全 0）

注意：本文件仅依赖你原项目已有的模块：
- utils.excel_utils.ExcelDataLoader
- problem.pure_seru.*（SeruFormation/SeruSchedule/Solution/Initialization/CalculateFitness）
- metaheuristics.common.ga_operator.GaOperator（用于 order_crossover/swap_mutation）
"""
import sys
import os
# 添加项目根目录到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import time
import math
import random
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from utils.excel_utils import ExcelDataLoader
from metaheuristics.common.ga_operator import GaOperator

from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.initialization import Initialization
from problem.pure_seru.calculate_fitness import CalculateFitness

from utils.config_loader import ConfigLoader


# ------------------------------
# 冲突模型（0~1 置信度）
# ------------------------------
class ConflictModel:
    """
    - worker_conf: 工人对置信度矩阵 C^W，shape=(W+1, W+1) 或 (W,W)（支持 1-based 与 0-based）
    - batch_conf : 批次对置信度矩阵 C^B，shape=(B+1, B+1) 或 (B,B)

    解释：C_ij ∈ [0,1]，越大表示“越不建议同 Seru”（但非绝对）。
    """
    def __init__(
        self,
        worker_conf: np.ndarray,
        batch_conf: np.ndarray,
        rho: float = 1.5,
        eps: float = 1e-6,
        alpha: float = 1.0,
        theta_hard_worker: Optional[float] = None,
        theta_hard_batch: Optional[float] = None,
    ):
        self.rho = float(rho)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.theta_hard_worker = theta_hard_worker  # 例如 0.98；None 表示不硬化
        self.theta_hard_batch = theta_hard_batch

        self.worker_conf = self._symmetrize_and_clip(worker_conf)
        self.batch_conf = self._symmetrize_and_clip(batch_conf)

        # 邻接表（稀疏化）——用于快速计算 Σ w_ij，避免 O(n^2)
        self.worker_adj = self._build_adj(self.worker_conf, eps=self.eps)
        self.batch_adj = self._build_adj(self.batch_conf, eps=self.eps)

        # W_tot 使用“基准置信度”计算（不含条件化），用于稳定控制 r*（10%）
        self.W_tot = self._total_weight(self.worker_adj) + self._total_weight(self.batch_adj)
        if self.W_tot <= 0:
            self.W_tot = 1.0  # 避免除零

        # 动态条件化（随 best_formation 更新）
        self._weakness_by_seru: List[Dict[Any, float]] = []  # seru_idx -> {product_type: weakness in [0,1]}

    @staticmethod
    def _symmetrize_and_clip(M: np.ndarray) -> np.ndarray:
        M = np.array(M, dtype=float)
        # clip
        M = np.clip(M, 0.0, 1.0)
        # symmetrize if square
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            M = 0.5 * (M + M.T)
            np.fill_diagonal(M, 0.0)
        return M

    @staticmethod
    def _build_adj(M: np.ndarray, eps: float) -> List[List[Tuple[int, float]]]:
        """
        返回 0-based 的邻接表；若输入为 1-based（尺寸 W+1），则 0 号会保留空边列表。
        """
        n = M.shape[0]
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            row = M[i]
            # 仅存 > eps 的边
            for j, w in enumerate(row):
                if j != i and w > eps:
                    adj[i].append((j, float(w)))
        return adj

    @staticmethod
    def _total_weight(adj: List[List[Tuple[int, float]]]) -> float:
        # 对称矩阵：总权重 = 1/2 Σ_i Σ_(j,w) w
        s = 0.0
        for i, nbrs in enumerate(adj):
            for j, w in nbrs:
                s += w
        return 0.5 * s

    @staticmethod
    def _logit(x: float, eps: float) -> float:
        x = min(max(x, eps), 1.0 - eps)
        return math.log(x / (1.0 - x))

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    # ------------------------------
    # 动态更新：基于 best_formation 构造 seru 的 weakness[type]
    # ------------------------------
    def update_dynamic_by_formation(self, best_formation: SeruFormation, config_seru, excel_loader: ExcelDataLoader):
        """
        计算每个 Seru 在各产品类型上的“弱势度 weakness ∈ [0,1]”。
        用于在调度阶段条件化批次对冲突权重 w_eff(s,p,q)。

        注：这里做的是“最小改动”的实现，不依赖你 CalculateFitness 内部的完整 g(i,t) 细节，
        但会尽可能复用 ExcelDataLoader 的 worker_to_product_dict 与 worker_to_task_dict。
        """
        # 收集产品类型集合
        product_types = set()
        w2p = getattr(excel_loader, "worker_to_product_dict", {}) or {}
        for wid, mp in w2p.items():
            for t in mp.keys():
                product_types.add(t)
        product_types = sorted(list(product_types))

        N = getattr(config_seru, "num_of_workers", None) or max(w2p.keys()) if w2p else 1
        max_multi = getattr(config_seru, "max_num_of_multiple_task", 0)

        w2task = getattr(excel_loader, "worker_to_task_dict", {}) or {}

        weakness_by_seru: List[Dict[Any, float]] = []
        for seru in getattr(best_formation, "seru_set", []):
            workers = getattr(seru, "workers_set", []) or []
            if len(workers) == 0 or len(product_types) == 0:
                weakness_by_seru.append({})
                continue

            # cap_s[t] = avg_i ( c_i * skill_{i,t} )
            cap: Dict[Any, float] = {}
            for t in product_types:
                s = 0.0
                for wid in workers:
                    # 系数 c_i
                    coeff = 0.0
                    if wid in w2task and isinstance(w2task[wid], dict):
                        coeff = float(w2task[wid].get("系数", 0.0) or 0.0)
                    c_i = 1.0 + coeff * (float(N) - float(max_multi))
                    # 熟练度
                    skill = 0.0
                    if wid in w2p and isinstance(w2p[wid], dict):
                        skill = float(w2p[wid].get(t, 0.0) or 0.0)
                    s += c_i * skill
                cap[t] = s / max(len(workers), 1)

            vals = list(cap.values())
            vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 0.0)
            denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0

            weakness: Dict[Any, float] = {}
            for t, v in cap.items():
                cap_norm = (v - vmin) / denom  # [0,1]
                weakness[t] = 1.0 - cap_norm   # [0,1]，越大越“弱”
            weakness_by_seru.append(weakness)

        self._weakness_by_seru = weakness_by_seru

    # ------------------------------
    # 条件化批次冲突权重：w_eff(s,p,q)
    # ------------------------------
    def batch_pair_weight_dynamic(
        self,
        seru_idx: int,
        base_conf: float,
        type_p: Any,
        type_q: Any,
    ) -> float:
        """
        w_eff(s,p,q) = sigmoid( logit( base_conf^rho ) + alpha*(pair_weak - 0.5) )
        pair_weak = 0.5*(weakness_s[type_p] + weakness_s[type_q])
        """
        # base_conf 是基准置信度（0~1）
        if base_conf <= self.eps:
            return 0.0
        base = float(base_conf) ** self.rho
        z = self._logit(base, eps=self.eps)

        if 0 <= seru_idx < len(self._weakness_by_seru):
            wk = self._weakness_by_seru[seru_idx]
            wp = float(wk.get(type_p, 0.5)) if wk else 0.5
            wq = float(wk.get(type_q, 0.5)) if wk else 0.5
            pair_weak = 0.5 * (wp + wq)
        else:
            pair_weak = 0.5

        z = z + self.alpha * (pair_weak - 0.5)
        return self._sigmoid(z)

    # ------------------------------
    # 计算违背量（P）与违背率（r）
    # ------------------------------
    def compute_worker_violation(self, formation: SeruFormation) -> float:
        """
        P_W：在每个 Seru 内对工人对 (i,j) 累加 C^W_ij（基准置信度）。
        使用邻接表加速。
        """
        serus = getattr(formation, "seru_set", []) or []
        P = 0.0

        # 构建 worker->seru_idx 映射
        worker_seru: Dict[int, int] = {}
        for s_idx, seru in enumerate(serus):
            for wid in getattr(seru, "workers_set", []) or []:
                worker_seru[int(wid)] = int(s_idx)

        # 遍历邻接表，避免双计数：仅统计 i<j
        for i, nbrs in enumerate(self.worker_adj):
            if i == 0:
                # 若你用 1-based 索引，0 行通常为空，这里跳过即可
                pass
            si = worker_seru.get(i, None)
            if si is None:
                continue
            for j, w in nbrs:
                if j <= i:
                    continue
                sj = worker_seru.get(j, None)
                if sj is None:
                    continue
                if si == sj:
                    P += float(w)
        return P

    def compute_batch_violation(
        self,
        formation: SeruFormation,
        schedule: SeruSchedule,
        excel_loader: ExcelDataLoader,
        mode: str = "base",
    ) -> float:
        """
        P_B：在每个 Seru 内对批次对 (p,q) 累加置信度权重。
        - mode="base"    ：使用基准矩阵 C^B（不条件化）
        - mode="dynamic" ：使用 w_eff(s,p,q)（需先 update_dynamic_by_formation(best_formation)）
        """
        serus = getattr(formation, "seru_set", []) or []
        S = len(serus)
        if S <= 0:
            return 0.0

        # 复制 CalculateFitness 的映射逻辑：第 i 个 assignment group -> seru i%S
        batches_assignment = getattr(schedule, "batches_assignment", None)
        if not batches_assignment:
            # 若没有 batches_assignment，则退化：使用 formation.seru_set[*].batches_set（若有）
            seru_batches = [list(getattr(seru, "batches_set", []) or []) for seru in serus]
        else:
            seru_batches = [[] for _ in range(S)]
            for i, group in enumerate(batches_assignment):
                seru_batches[i % S].extend(list(group))

        # 批次 -> 产品类型
        b2p = getattr(excel_loader, "batch_to_product_dict", {}) or {}
        def batch_type(bid: int):
            info = b2p.get(bid, {})
            # 兼容字段名
            return info.get("产品类型", info.get("product_type", None))

        P = 0.0
        # 对每个 seru，建立 set 方便 membership
        for s_idx, blist in enumerate(seru_batches):
            if not blist:
                continue
            bset = set(int(b) for b in blist)

            # 遍历邻接表的边，只对同 seru 的 (p,q) 计入
            for p in bset:
                for q, c_pq in self.batch_adj[p]:
                    q = int(q)
                    if q <= p:
                        continue
                    if q not in bset:
                        continue

                    if mode == "base":
                        P += float(c_pq)
                    else:
                        tp = batch_type(p)
                        tq = batch_type(q)
                        w_eff = self.batch_pair_weight_dynamic(
                            seru_idx=s_idx,
                            base_conf=float(c_pq),
                            type_p=tp,
                            type_q=tq,
                        )
                        # 可选：极高置信硬化（只在 dynamic 模式下做更有意义）
                        if self.theta_hard_batch is not None and w_eff >= self.theta_hard_batch:
                            # 这里不直接判 infeasible（否则需要回传到解码器）；
                            # 在“最小改动”版中用一个大罚项来近似硬禁止。
                            P += 10.0 * float(w_eff)
                        else:
                            P += float(w_eff)
        return P

    def compute_r(
        self,
        formation: SeruFormation,
        schedule: SeruSchedule,
        excel_loader: ExcelDataLoader,
        batch_mode: str,
    ) -> Tuple[float, float, float]:
        """
        返回：r, P_W, P_B（其中 P_B 可为 dynamic）
        """
        P_W = self.compute_worker_violation(formation)
        P_B = self.compute_batch_violation(formation, schedule, excel_loader, mode=batch_mode)
        r = (P_W + P_B) / float(self.W_tot)
        return float(r), float(P_W), float(P_B)


def _load_conf_matrix(path: Optional[str], size: int, one_based: bool = True) -> np.ndarray:
    """
    加载 0~1 置信度矩阵：支持 .npy / .csv / .xlsx。
    若 path 为空，返回全 0 矩阵（无先验）。
    """
    if not path:
        n = size + 1 if one_based else size
        return np.zeros((n, n), dtype=float)

    if path.endswith(".npy"):
        M = np.load(path)
    elif path.endswith(".csv"):
        M = np.loadtxt(path, delimiter=",")
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        # 默认读取第一个 sheet；如需指定 sheet，可在 path 中用 "file.xlsx::Sheet1"
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
    # 若尺寸不匹配，尝试 pad/crop（最小改动：尽量不报错）
    n = size + 1 if one_based else size
    if M.shape[0] != n or M.shape[1] != n:
        M2 = np.zeros((n, n), dtype=float)
        r = min(n, M.shape[0]); c = min(n, M.shape[1])
        M2[:r, :c] = M[:r, :c]
        M = M2
    return M


# ------------------------------
# 最小改动版 CCEA 主类
# ------------------------------
class CCEA:
    def __init__(self):
        # 加载配置
        self.config_seru = ConfigLoader.get_config('config_seru')
        self.config_ccea = ConfigLoader.get_config('config_ccea')

        # 读取基础数据
        self.loader = ExcelDataLoader()
        self.loader.read_data(excel_path=self.config_seru.seru_data_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.due_dates_path, config_sheet=self.config_seru)
        self.loader.read_data(excel_path=self.config_seru.batch_types_path, config_sheet=self.config_seru)

        # 冲突矩阵配置（路径/参数）
        W = int(getattr(self.config_seru, "num_of_workers", 0) or 0)
        B = int(getattr(self.config_seru, "num_of_batches", 0) or 0)

        worker_path = getattr(self.config_seru, "worker_confidence_path", None) \
                      or getattr(self.config_seru, "worker_conf_path", None) \
                      or getattr(self.config_ccea, "worker_confidence_path", None)
        batch_path = getattr(self.config_seru, "batch_confidence_path", None) \
                     or getattr(self.config_seru, "batch_conf_path", None) \
                     or getattr(self.config_ccea, "batch_confidence_path", None)

        rho = float(getattr(self.config_ccea, "conflict_rho", 1.5))
        eps = float(getattr(self.config_ccea, "conflict_eps", 1e-6))
        alpha = float(getattr(self.config_ccea, "dynamic_conflict_alpha", 1.0))
        thw = getattr(self.config_ccea, "theta_hard_worker", None)
        thb = getattr(self.config_ccea, "theta_hard_batch", None)
        thw = float(thw) if thw is not None else None
        thb = float(thb) if thb is not None else None

        worker_conf = _load_conf_matrix(worker_path, size=W, one_based=True)
        batch_conf = _load_conf_matrix(batch_path, size=B, one_based=True)

        self.conflict_model = ConflictModel(
            worker_conf=worker_conf,
            batch_conf=batch_conf,
            rho=rho,
            eps=eps,
            alpha=alpha,
            theta_hard_worker=thw,
            theta_hard_batch=thb,
        )

        # 软约束参数（10% 容忍度控制）
        self.r_target = float(getattr(self.config_ccea, "conflict_r_target", 0.10))
        self.eta = float(getattr(self.config_ccea, "conflict_eta", 0.1))
        self.gamma = float(getattr(self.config_ccea, "conflict_gamma_init", 1.0))
        self.gamma_min = float(getattr(self.config_ccea, "conflict_gamma_min", 0.0))
        self.gamma_max = float(getattr(self.config_ccea, "conflict_gamma_max", 5.0))

        # 局部搜索强度（最小改动：用 swap 邻域）
        self.ls_iters = int(getattr(self.config_ccea, "local_search_iters", 30))

    # ------------------------------
    # 工具：重置 seru 指标，避免多次 evaluate 的状态累积
    # ------------------------------
    @staticmethod
    def _reset_seru_metrics(formation: SeruFormation):
        for seru in getattr(formation, "seru_set", []) or []:
            # 这些字段在你的实体里存在（若不存在，setattr 不会报错）
            for attr in ["throughput_time", "processing_time", "labour_time", "tardiness", "fitness"]:
                if hasattr(seru, attr):
                    setattr(seru, attr, 0.0)
            # 重要：避免 batches_set 被多次 extend 造成累积
            if hasattr(seru, "batches_set"):
                setattr(seru, "batches_set", [])

    def _update_gamma(self, r_avg: float):
        # γ_{t+1} = clip( γ_t * exp( η (r_avg - r*) ) )
        g = self.gamma * math.exp(self.eta * (float(r_avg) - self.r_target))
        self.gamma = max(self.gamma_min, min(self.gamma_max, g))

    def _evaluate_solution(self, solution: Solution, batch_mode: str, M_ref: float) -> float:
        """
        评估一个 solution：
        1) 用原 CalculateFitness 得到 makespan（真实目标）
        2) 计算 r（工人 P_W + 批次 P_B，其中批次可 dynamic）
        3) solution.fitness = makespan + gamma*M_ref*r
           同步写回 formation.fitness / schedule.fitness，确保 selection 生效
        返回：r（用于统计 r_avg）
        """
        self._reset_seru_metrics(solution.formation)
        CalculateFitness.calculate_fitness(solution=solution, config_seru=self.config_seru)

        M = float(getattr(solution, "makespan", None) or getattr(solution, "fitness", 0.0) or 0.0)
        # 若 CalculateFitness 内已将 solution.fitness 覆盖为 makespan，可用 makespan 作为基准
        excel_loader = ExcelDataLoader.instance()

        r, P_W, P_B = self.conflict_model.compute_r(
            formation=solution.formation,
            schedule=solution.schedule,
            excel_loader=excel_loader,
            batch_mode=batch_mode,
        )

        penalized = M + self.gamma * float(M_ref) * float(r)

        # 回写
        solution.makespan = M
        solution.fitness = penalized
        if hasattr(solution, "formation") and hasattr(solution.formation, "fitness"):
            solution.formation.fitness = penalized
        if hasattr(solution, "schedule") and hasattr(solution.schedule, "fitness"):
            solution.schedule.fitness = penalized

        # 可选：保留调试字段
        setattr(solution, "_conflict_r", r)
        setattr(solution, "_conflict_PW", P_W)
        setattr(solution, "_conflict_PB", P_B)
        return r

    # ------------------------------
    # 选择（用 fitness）
    # ------------------------------
    @staticmethod
    def _tournament_selection(population: List[Any]) -> Any:
        a, b = random.sample(population, 2)
        fa = float(getattr(a, "fitness", float("inf")))
        fb = float(getattr(b, "fitness", float("inf")))
        return a if fa <= fb else b

    # ------------------------------
    # 局部搜索（最小改动：swap 邻域 + 重评估）
    # ------------------------------
    def _local_search_formation(self, formation: SeruFormation, best_schedule: SeruSchedule, M_ref: float) -> SeruFormation:
        best = formation
        best_sol = Solution(formation=best, schedule=best_schedule)
        best_r = self._evaluate_solution(best_sol, batch_mode="base", M_ref=M_ref)
        best_fit = float(best.fitness)

        for _ in range(self.ls_iters):
            nb = best.__copy__()
            if not getattr(nb, "formation_code", None) or len(nb.formation_code) < 2:
                break
            i, j = random.sample(range(len(nb.formation_code)), 2)
            nb.formation_code[i], nb.formation_code[j] = nb.formation_code[j], nb.formation_code[i]
            Initialization.produce_seru_formation(self.config_seru.num_of_workers, nb)

            sol = Solution(formation=nb, schedule=best_schedule)
            self._evaluate_solution(sol, batch_mode="base", M_ref=M_ref)
            if float(nb.fitness) < best_fit:
                best, best_fit = nb, float(nb.fitness)

        return best

    def _local_search_schedule(self, schedule: SeruSchedule, best_formation: SeruFormation, M_ref: float) -> SeruSchedule:
        best = schedule
        sol0 = Solution(formation=best_formation, schedule=best)
        self._evaluate_solution(sol0, batch_mode="dynamic", M_ref=M_ref)
        best_fit = float(best.fitness)

        for _ in range(self.ls_iters):
            nb = best.__copy__()
            if not getattr(nb, "schedule_code", None) or len(nb.schedule_code) < 2:
                break
            i, j = random.sample(range(len(nb.schedule_code)), 2)
            nb.schedule_code[i], nb.schedule_code[j] = nb.schedule_code[j], nb.schedule_code[i]
            Initialization.produce_seru_schedule(self.config_seru.num_of_batches, nb)

            sol = Solution(formation=best_formation, schedule=nb)
            self._evaluate_solution(sol, batch_mode="dynamic", M_ref=M_ref)
            if float(nb.fitness) < best_fit:
                best, best_fit = nb, float(nb.fitness)

        return best

    # ------------------------------
    # 主流程
    # ------------------------------
    def run(self) -> (SeruFormation, SeruSchedule, Solution):
        # 1) 初始化 PSF / PSS
        PSF: List[SeruFormation] = []
        for _ in range(self.config_ccea.population_size):
            formation_code = Initialization.initial_formation_code(self.config_seru.num_of_workers)
            formation = SeruFormation(formation_code=formation_code)
            Initialization.produce_seru_formation(self.config_seru.num_of_workers, formation)
            PSF.append(formation)

        PSS: List[SeruSchedule] = []
        for _ in range(self.config_ccea.population_size):
            schedule_code = Initialization.initial_schedule_code(self.config_seru.num_of_workers, self.config_seru.num_of_batches)
            schedule = SeruSchedule(schedule_code=schedule_code)
            Initialization.produce_seru_schedule(self.config_seru.num_of_batches, schedule)
            PSS.append(schedule)

        # 2) 初始化 best 解（先用 base 模式评估一次）
        best_formation = PSF[0]
        best_schedule = PSS[0]

        # 先初始化 dynamic（用 best_formation）
        self.conflict_model.update_dynamic_by_formation(best_formation, self.config_seru, ExcelDataLoader.instance())

        current_best_solution = Solution(formation=best_formation, schedule=best_schedule)
        # M_ref 初始：用 makespan 先跑一次
        self._reset_seru_metrics(current_best_solution.formation)
        CalculateFitness.calculate_fitness(solution=current_best_solution, config_seru=self.config_seru)
        M_ref = float(getattr(current_best_solution, "makespan", 1.0) or 1.0)

        self._evaluate_solution(current_best_solution, batch_mode="dynamic", M_ref=M_ref)
        best_solution = current_best_solution

        # 3) 主循环
        start_time = time.time()
        iteration = 0
        while time.time() - start_time < self.config_ccea.max_runtime:
            iteration += 1

            # 用当前 best makespan 作为参考尺度（也可用均值）
            M_ref = float(getattr(best_solution, "makespan", 1.0) or 1.0)

            # ------------------------------
            # 3.1 进化 PSF（固定 best_schedule；batch_mode=base）
            # ------------------------------
            r_list = []
            for formation in PSF:
                sol = Solution(formation=formation, schedule=best_schedule)
                r = self._evaluate_solution(sol, batch_mode="base", M_ref=M_ref)
                r_list.append(r)

            if r_list:
                self._update_gamma(float(sum(r_list) / len(r_list)))

            selected = [self._tournament_selection(PSF) for _ in range(len(PSF))]

            offspring: List[SeruFormation] = []
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i + 1]
                c1, c2 = p1.__copy__(), p2.__copy__()

                c1.formation_code, c2.formation_code = GaOperator.order_crossover(
                    p1.formation_code, p2.formation_code, self.config_ccea
                )
                c1.formation_code = GaOperator.swap_mutation(c1.formation_code, self.config_ccea)
                c2.formation_code = GaOperator.swap_mutation(c2.formation_code, self.config_ccea)

                Initialization.produce_seru_formation(self.config_seru.num_of_workers, c1)
                Initialization.produce_seru_formation(self.config_seru.num_of_workers, c2)
                offspring.extend([c1, c2])

            PSF = offspring
            best_formation = min(PSF, key=lambda x: float(getattr(x, "fitness", float("inf"))))

            # 局部搜索（最小改动 swap）
            best_formation = self._local_search_formation(best_formation, best_schedule, M_ref=M_ref)

            # ------------------------------
            # 3.2 动态更新：根据当前最优构造 best_formation 更新调度冲突模型
            # ------------------------------
            self.conflict_model.update_dynamic_by_formation(best_formation, self.config_seru, ExcelDataLoader.instance())

            # ------------------------------
            # 3.3 进化 PSS（固定 best_formation；batch_mode=dynamic）
            # ------------------------------
            r_list = []
            for schedule in PSS:
                sol = Solution(formation=best_formation, schedule=schedule)
                r = self._evaluate_solution(sol, batch_mode="dynamic", M_ref=M_ref)
                r_list.append(r)

            if r_list:
                self._update_gamma(float(sum(r_list) / len(r_list)))

            selected = [self._tournament_selection(PSS) for _ in range(len(PSS))]

            offspring: List[SeruSchedule] = []
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i + 1]
                c1, c2 = p1.__copy__(), p2.__copy__()

                c1.schedule_code, c2.schedule_code = GaOperator.order_crossover(
                    p1.schedule_code, p2.schedule_code, self.config_ccea
                )
                c1.schedule_code = GaOperator.swap_mutation(c1.schedule_code, self.config_ccea)
                c2.schedule_code = GaOperator.swap_mutation(c2.schedule_code, self.config_ccea)

                Initialization.produce_seru_schedule(self.config_seru.num_of_batches, c1)
                Initialization.produce_seru_schedule(self.config_seru.num_of_batches, c2)
                offspring.extend([c1, c2])

            PSS = offspring
            best_schedule = min(PSS, key=lambda x: float(getattr(x, "fitness", float("inf"))))

            # 局部搜索（最小改动 swap）
            best_schedule = self._local_search_schedule(best_schedule, best_formation, M_ref=M_ref)

            # ------------------------------
            # 3.4 更新当前解 / 全局最优
            # ------------------------------
            current_best_solution = Solution(formation=best_formation, schedule=best_schedule)
            self._evaluate_solution(current_best_solution, batch_mode="dynamic", M_ref=M_ref)

            if float(getattr(current_best_solution, "fitness", float("inf"))) < float(getattr(best_solution, "fitness", float("inf"))):
                best_solution = current_best_solution

            if iteration % 1 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration} | fitness={best_solution.fitness:.6f} | makespan={best_solution.makespan:.6f} | "
                      f"r={getattr(best_solution,'_conflict_r',0.0):.4f} | gamma={self.gamma:.4f} | t={elapsed:.2f}s")

        return best_solution.formation, best_solution.schedule, best_solution


def main():
    ConfigLoader.preload_all()
    ccea = CCEA()
    best_formation, best_schedule, best_solution = ccea.run()

    print("Best Formation:", best_formation)
    if hasattr(best_formation, "seru_set"):
        for i, seru in enumerate(best_formation.seru_set, start=1):
            print(f"Seru {i}: {len(getattr(seru, 'workers_set', []) or [])} workers")

    print("Best Scheduling:", best_schedule)
    print("Best Solution:", best_solution)


if __name__ == "__main__":
    main()
