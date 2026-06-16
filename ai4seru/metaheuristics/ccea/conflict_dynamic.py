# -*- coding: utf-8 -*-
"""
conflict_dynamic.py
冲突模型（0~1 置信度）+ “调度冲突随 best_formation 动态更新”的条件化权重 w_eff

设计要点：
- worker_conf 与 batch_conf 都是 0~1 的置信度矩阵：越大越“不建议同 Seru”，但不是硬约束。
- r（违背率）建议使用“基准矩阵”计算，便于 gamma 自适应稳定控制 10% 容忍度；
  解码器决策时可以使用动态条件化后的 w_eff（仅影响构造/调度的贪心选择）。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np


def sym_clip(M: np.ndarray) -> np.ndarray:
    M = np.array(M, dtype=float)
    M = np.clip(M, 0.0, 1.0)
    if M.ndim == 2 and M.shape[0] == M.shape[1]:
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 0.0)
    return M


def build_adj(M: np.ndarray, eps: float) -> List[List[Tuple[int, float]]]:
    n = M.shape[0]
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        row = M[i]
        for j, w in enumerate(row):
            if j != i and w > eps:
                adj[i].append((j, float(w)))
    return adj


def total_weight(adj: List[List[Tuple[int, float]]]) -> float:
    s = 0.0
    for nbrs in adj:
        for _, w in nbrs:
            s += w
    return 0.5 * s  # 对称矩阵去双计数


def logit(x: float, eps: float) -> float:
    x = min(max(x, eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class DynamicSeruProfile:
    """best_formation 条件化后，每个 Seru 的弱势度 weakness[type] ∈ [0,1]"""
    weakness_by_type: Dict[Any, float]


class ConflictModel:
    """
    worker_conf/batch_conf: 0~1 置信度矩阵（支持 0-based 或 1-based；推荐与你项目保持 1-based）
    rho: 置信度锐化指数（>1 强化高置信、弱化低置信）
    alpha: Seru 能力结构对批次冲突权重的影响强度
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
        self.theta_hard_worker = theta_hard_worker
        self.theta_hard_batch = theta_hard_batch

        self.worker_conf = sym_clip(worker_conf)
        self.batch_conf = sym_clip(batch_conf)

        self.worker_adj = build_adj(self.worker_conf, eps=self.eps)
        self.batch_adj = build_adj(self.batch_conf, eps=self.eps)

        self.W_tot = total_weight(self.worker_adj) + total_weight(self.batch_adj)
        if self.W_tot <= 0:
            self.W_tot = 1.0

        # best_formation 条件化的 Seru profile（按 seru_idx 存）
        self.seru_profiles: List[DynamicSeruProfile] = []

    # -------------------------
    # r（违背率）计算：基准矩阵
    # -------------------------
    def worker_violation_base(self, worker_seru: Dict[int, int]) -> float:
        P = 0.0
        for i, nbrs in enumerate(self.worker_adj):
            si = worker_seru.get(i)
            if si is None:
                continue
            for j, w in nbrs:
                if j <= i:
                    continue
                sj = worker_seru.get(j)
                if sj is None:
                    continue
                if si == sj:
                    P += w
        return P

    def batch_violation_base(self, batch_seru: Dict[int, int]) -> float:
        P = 0.0
        for p, nbrs in enumerate(self.batch_adj):
            sp = batch_seru.get(p)
            if sp is None:
                continue
            for q, w in nbrs:
                if q <= p:
                    continue
                sq = batch_seru.get(q)
                if sq is None:
                    continue
                if sp == sq:
                    P += w
        return P

    # -------------------------
    # 动态条件化：w_eff(s,p,q)
    # -------------------------
    def update_seru_profiles(self, seru_caps_by_type: List[Dict[Any, float]]):
        """
        输入：seru_caps_by_type[s][t] = cap（越大越强）
        输出：seru_profiles[s].weakness_by_type[t] = weakness ∈ [0,1]
        """
        profiles: List[DynamicSeruProfile] = []
        for cap in seru_caps_by_type:
            if not cap:
                profiles.append(DynamicSeruProfile(weakness_by_type={}))
                continue
            vals = list(cap.values())
            vmin, vmax = min(vals), max(vals)
            denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
            weakness = {}
            for t, v in cap.items():
                cap_norm = (float(v) - vmin) / denom  # [0,1]
                weakness[t] = 1.0 - cap_norm          # [0,1]
            profiles.append(DynamicSeruProfile(weakness_by_type=weakness))
        self.seru_profiles = profiles

    def w_eff_batch_pair(self, seru_idx: int, base_conf: float, type_p: Any, type_q: Any) -> float:
        """
        w_eff(s,p,q) = sigmoid( logit( (base_conf)^rho ) + alpha*(pair_weak - 0.5) )
        pair_weak = 0.5*(weakness_s[type_p] + weakness_s[type_q])
        """
        if base_conf <= self.eps:
            return 0.0
        base = float(base_conf) ** self.rho
        z = logit(base, eps=self.eps)

        pair_weak = 0.5
        if 0 <= seru_idx < len(self.seru_profiles):
            wk = self.seru_profiles[seru_idx].weakness_by_type
            if wk:
                wp = float(wk.get(type_p, 0.5))
                wq = float(wk.get(type_q, 0.5))
                pair_weak = 0.5 * (wp + wq)

        z = z + self.alpha * (pair_weak - 0.5)
        w = sigmoid(z)
        return w
