# -*- coding: utf-8 -*-
"""
rk_decoders.py
Random Keys → 解码器：
- decode_formation：工人优先级 → Seru 构造（K 个 Seru）
- decode_schedule ：批次优先级 → 批次分配/排序（K 个 Seru 对应 K 个 assignment groups）

实现目标：在解码时把“冲突矩阵 + 动态 w_eff”前移到决策里，减少无效解评估，提升收敛速度。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from metaheuristics.ccea.conflict_dynamic import ConflictModel


def _safe_get(d: Dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def extract_batch_type_and_qty(batch_id: int, batch_to_product: Dict[int, Dict[str, Any]]) -> Tuple[Any, float]:
    info = batch_to_product.get(int(batch_id), {}) if batch_to_product else {}
    t = _safe_get(info, "产品类型", "product_type", default=None)
    qty = _safe_get(info, "批次规模", "批次数量", "quantity", "size", default=1.0)
    try:
        qty = float(qty)
    except Exception:
        qty = 1.0
    return t, qty


def compute_seru_caps_by_type(
    seru_workers: List[List[int]],
    worker_to_product: Dict[int, Dict[Any, float]],
    worker_to_task: Dict[int, Dict[str, Any]],
    num_workers: int,
    max_multiple_task: float,
) -> List[Dict[Any, float]]:
    """
    cap_s[t] = sum_{i in seru} (c_i * skill_{i,t})，用于估计 seru 对不同类型的“能力”
    c_i = 1 + coeff*(N - max_multiple_task)，与项目 calculate_fitness 中的形式兼容（但这里做的是轻量 proxy）。
    """
    # 收集所有类型
    types = set()
    for wid, mp in (worker_to_product or {}).items():
        for t in mp.keys():
            types.add(t)
    types = list(types)

    caps: List[Dict[Any, float]] = []
    for workers in seru_workers:
        cap = {}
        for t in types:
            s = 0.0
            for wid in workers:
                mp = worker_to_product.get(wid, {}) if worker_to_product else {}
                skill = float(mp.get(t, 0.0) or 0.0)

                coeff = 0.0
                mt = worker_to_task.get(wid, {}) if worker_to_task else {}
                try:
                    coeff = float(mt.get("系数", 0.0) or 0.0)
                except Exception:
                    coeff = 0.0
                c_i = 1.0 + coeff * (float(num_workers) - float(max_multiple_task))
                s += c_i * skill
            cap[t] = s
        caps.append(cap)
    return caps


def decode_formation(
    keys_workers: np.ndarray,
    num_serus: int,
    conflict: ConflictModel,
    worker_to_product: Dict[int, Dict[Any, float]],
    worker_to_task: Dict[int, Dict[str, Any]],
    num_workers: int,
    max_multiple_task: float,
    gamma: float,
    M_ref: float,
) -> List[List[int]]:
    """
    解码输出：seru_workers[s] = [worker_id, ...]（worker_id 建议 1-based，与你项目一致）

    决策规则（图感知贪心）：
    对优先级顺序中的每个工人 v，选择 Seru s 使 Δcost 最小：
        Δcost = ΔM_proxy + gamma * M_ref * (ΔP / W_tot)

    其中：
    - ΔP：新增工人冲突（基准矩阵 C^W，经 rho 锐化后用于决策更尖锐）
    - ΔM_proxy：为了最小可运行，这里用“Seru 规模均衡 + 能力均衡”的代理（不调用 CalculateFitness）
    """
    W = len(keys_workers)
    order = np.argsort(keys_workers)  # 0-based index in [0,W-1]
    workers = [int(i + 1) for i in order]  # 转成 1-based worker_id

    K = int(max(1, num_serus))
    seru_workers: List[List[int]] = [[] for _ in range(K)]

    # 先确保每个 Seru 至少 1 人（避免空 seru 导致后续 cap 不稳定）
    for s in range(K):
        if s < len(workers):
            seru_workers[s].append(workers[s])
    remaining = workers[K:]

    # 预计算：用于 ΔM_proxy 的均衡项
    def size_penalty(s: int) -> float:
        sizes = [len(x) for x in seru_workers]
        avg = sum(sizes) / float(K)
        return (len(seru_workers[s]) + 1 - avg) ** 2

    # 冲突增量（使用 worker_conf^rho）
    def conflict_delta(s: int, v: int) -> float:
        dP = 0.0
        # 使用邻接表快速；为简单起见，这里直接扫已有成员
        for u in seru_workers[s]:
            # 置信度矩阵可能是 1-based（0 行空），因此直接用 u,v 索引是可行的
            base = float(conflict.worker_conf[u][v]) if u < conflict.worker_conf.shape[0] and v < conflict.worker_conf.shape[1] else 0.0
            dP += (base ** conflict.rho)
        return dP

    # 能力均衡 proxy：新增 v 后该 seru 的“cap 总和”变化越大越好（因此以 -cap_gain 作为成本）
    # 为最小可运行：只用 v 的技能向量总和近似 cap_gain
    def cap_gain_proxy(v: int) -> float:
        mp = worker_to_product.get(v, {}) if worker_to_product else {}
        return float(sum(float(x or 0.0) for x in mp.values()))

    for v in remaining:
        best_s = 0
        best_val = float("inf")
        gain = cap_gain_proxy(v)

        for s in range(K):
            dP = conflict_delta(s, v)
            dR = dP / float(conflict.W_tot)
            dM = size_penalty(s) - 0.01 * gain  # 0.01 是一个温和的缩放，避免 gain 主导
            val = dM + float(gamma) * float(M_ref) * float(dR)
            if val < best_val:
                best_val = val
                best_s = s

        seru_workers[best_s].append(v)

    return seru_workers


def decode_schedule(
    keys_ins: np.ndarray,
    keys_seq: Optional[np.ndarray],
    seru_workers: List[List[int]],
    conflict: ConflictModel,
    batch_to_product: Dict[int, Dict[str, Any]],
    worker_to_product: Dict[int, Dict[Any, float]],
    worker_to_task: Dict[int, Dict[str, Any]],
    num_workers: int,
    max_multiple_task: float,
    gamma: float,
    M_ref: float,
    use_dynamic: bool,
) -> List[List[int]]:
    """
    解码输出：batches_assignment[s] = [batch_id, ...]（长度 K，对应 K 个 seru）

    决策规则：对每个批次 b（按 keys_ins 排序），选择 Seru s 使：
        Δcost = ΔM_proxy + gamma*M_ref*(ΔP/W_tot)
    - ΔP：批次冲突增量；若 use_dynamic=True，则用 w_eff(s,p,b)；否则用 base_conf^rho
    - ΔM_proxy：用 proxy load（工作量/能力）近似 makespan 增量
    """
    K = len(seru_workers)
    B = len(keys_ins)
    order = np.argsort(keys_ins)
    batches = [int(i + 1) for i in order]  # 1-based batch_id

    # 先计算 seru 的能力向量 cap_s[t]
    caps = compute_seru_caps_by_type(
        seru_workers=seru_workers,
        worker_to_product=worker_to_product,
        worker_to_task=worker_to_task,
        num_workers=num_workers,
        max_multiple_task=max_multiple_task,
    )
    # 更新动态 profile（weakness）
    conflict.update_seru_profiles(caps)

    # 代理负载：load_s 累积 Σ (qty / cap_s[type])
    loads = [0.0 for _ in range(K)]
    assign: List[List[int]] = [[] for _ in range(K)]

    # 批次对冲突增量：与已在 s 中的批次发生冲突
    def conflict_delta_batch(s: int, b: int) -> float:
        tb, _ = extract_batch_type_and_qty(b, batch_to_product)
        dP = 0.0
        for p in assign[s]:
            tp, _ = extract_batch_type_and_qty(p, batch_to_product)
            base = float(conflict.batch_conf[p][b]) if p < conflict.batch_conf.shape[0] and b < conflict.batch_conf.shape[1] else 0.0
            if use_dynamic:
                w = conflict.w_eff_batch_pair(seru_idx=s, base_conf=base, type_p=tp, type_q=tb)
                dP += float(w)
            else:
                dP += (base ** conflict.rho)
        return dP

    def load_increment(s: int, b: int) -> float:
        t, qty = extract_batch_type_and_qty(b, batch_to_product)
        cap = float(caps[s].get(t, 0.0) or 0.0)
        # cap 越大越快；cap=0 表示不可做/极慢
        unit = (qty / cap) if cap > 1e-9 else (qty * 1e6)
        old_max = max(loads)
        new_load_s = loads[s] + unit
        new_max = max(old_max, new_load_s)
        return new_max - old_max

    for b in batches:
        best_s = 0
        best_val = float("inf")
        for s in range(K):
            dP = conflict_delta_batch(s, b)
            dR = dP / float(conflict.W_tot)
            dM = load_increment(s, b)
            val = dM + float(gamma) * float(M_ref) * float(dR)
            if val < best_val:
                best_val = val
                best_s = s
        assign[best_s].append(b)
        # update proxy load
        t, qty = extract_batch_type_and_qty(b, batch_to_product)
        cap = float(caps[best_s].get(t, 0.0) or 0.0)
        unit = (qty / cap) if cap > 1e-9 else (qty * 1e6)
        loads[best_s] += unit

    # seru 内顺序：如果有 keys_seq，则按 keys_seq 排序
    if keys_seq is not None:
        seq = keys_seq
        for s in range(K):
            assign[s].sort(key=lambda bid: float(seq[bid - 1]))

    return assign
