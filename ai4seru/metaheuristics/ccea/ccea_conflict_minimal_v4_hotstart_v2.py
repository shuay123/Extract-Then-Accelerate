# -*- coding: utf-8 -*-
"""
最终修复版 V4：CCEA + 冲突软约束 (Fully Symmetric Logic)
更新内容：
1. 工人动态权重计算逻辑与批次完全对齐：引入 Seru 平均 Weakness 作为调节因子。
   公式：Sigmoid( Logit(w) + alpha * (Seru_Avg_Weakness - 0.5) )
2. 保持对称更新策略：
   - 进化构造: Worker=Dynamic, Batch=Base
   - 进化调度: Worker=Base, Batch=Dynamic
"""
import sys
import os
import time
import math
import random
import copy
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 路径处理
# ---------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.excel_utils import ExcelDataLoader
from metaheuristics.common.ga_operatorv2 import GaOperator
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution
from problem.pure_seru.initialization import Initialization
from problem.pure_seru.calculate_fitness import CalculateFitness
from utils.config_loader import ConfigLoader


# ------------------------------
# 冲突模型（核心逻辑）
# ------------------------------
class ConflictModel:
    def __init__(
        self,
        worker_conf: np.ndarray,
        batch_conf: np.ndarray,
        rho: float = 1.5,
        eps: float = 1e-6,
        alpha: float = 1.0,
        theta_hard_worker: Optional[float] = None,
        theta_hard_batch: Optional[float] = None,
        # [新增参数] 仅保留每行最大的 Top-K 个冲突，防止稠密矩阵卡死
        top_k_worker: int = 10, 
        top_k_batch: int = 20    
    ):
        self.rho = float(rho)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.theta_hard_worker = theta_hard_worker
        self.theta_hard_batch = theta_hard_batch

        self.worker_conf = self._symmetrize_and_clip(worker_conf)
        self.batch_conf = self._symmetrize_and_clip(batch_conf)

        # [优化1] 构建邻接表时进行 Top-K 截断
        # 即使 GNN 给出全连接，我们只关心每个人/批次最严重的 K 个冲突
        self.worker_adj = self._build_adj(self.worker_conf, eps=self.eps, top_k=top_k_worker)
        self.batch_adj = self._build_adj(self.batch_conf, eps=self.eps, top_k=top_k_batch)
        
        # [优化2] 预计算静态部分的 Logit 值
        # 原始公式: Sigmoid( Logit(w^rho) + alpha * ... )
        # Logit(w^rho) 只和 w 有关，可以预先算好存入邻接表
        self._precompute_logits(self.worker_adj)
        self._precompute_logits(self.batch_adj)

        self.W_tot = self._total_weight(self.worker_adj) + self._total_weight(self.batch_adj)
        if self.W_tot <= 1e-9:
            self.W_tot = 1.0  

        self._weakness_by_seru: List[Dict[Any, float]] = []

    def _precompute_logits(self, adj: List[List[Tuple[int, float, float]]]):
        """
        原地修改 adj，增加第三个字段：precomputed_logit
        adj 结构变为: [(neighbor_idx, weight, logit_value), ...]
        """
        for i in range(len(adj)):
            new_list = []
            for j, w in adj[i]:
                # 预先计算 Logit(w^rho)
                base = float(w) ** self.rho
                logit_val = self._logit(base, self.eps)
                new_list.append((j, w, logit_val))
            adj[i] = new_list

    @staticmethod
    def _symmetrize_and_clip(M: np.ndarray) -> np.ndarray:
        M = np.array(M, dtype=float)
        M = np.clip(M, 0.0, 1.0)
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            M = 0.5 * (M + M.T)
            np.fill_diagonal(M, 0.0)
        return M

    @staticmethod
    def _build_adj(M: np.ndarray, eps: float, top_k: int) -> List[List[Tuple[int, float]]]:
        """
        [关键优化] 引入 Top-K 截断。
        """
        n = M.shape[0]
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            row = M[i]
            # 找出大于 eps 的索引
            valid_idxs = np.where(row > eps)[0]
            # 过滤掉自己
            valid_idxs = valid_idxs[valid_idxs != i]
            
            if len(valid_idxs) == 0:
                continue
                
            # 如果邻居数量超过 Top-K，只取权重最大的 K 个
            if len(valid_idxs) > top_k:
                # argsort 返回从小到大的索引，取最后 K 个并反转
                # 注意这里是对 valid_idxs 对应的 weights 排序
                w_sub = row[valid_idxs]
                top_k_args = np.argsort(w_sub)[-top_k:]
                best_idxs = valid_idxs[top_k_args]
            else:
                best_idxs = valid_idxs
            
            for j in best_idxs:
                adj[i].append((int(j), float(row[j])))
        return adj

    @staticmethod
    def _total_weight(adj) -> float:
        s = 0.0
        for i, nbrs in enumerate(adj):
            # nbrs 现在是 (j, w, logit)
            for item in nbrs:
                s += item[1] # item[1] is w
        return 0.5 * s

    @staticmethod
    def _logit(x: float, eps: float) -> float:
        # 加上容错，防止 log(0)
        x = min(max(x, 1e-9), 1.0 - 1e-9)
        return math.log(x / (1.0 - x))

    @staticmethod
    def _sigmoid(z: float) -> float:
        # 防止 exp 溢出
        if z > 100: return 1.0
        if z < -100: return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def update_dynamic_by_formation(self, best_formation: SeruFormation, config_seru, excel_loader: ExcelDataLoader):
        # ... (保持原样，未修改) ...
        """计算 Seru 在各产品类型上的 Weakness"""
        product_types = set()
        w2p = getattr(excel_loader, "worker_to_product_dict", {}) or {}
        if not w2p: return 

        for mp in w2p.values():
            for t in mp.keys():
                product_types.add(t)
        product_types = sorted(list(product_types))
        
        N = getattr(config_seru, "num_of_workers", 0)
        if N == 0 and w2p: N = max(w2p.keys())
        max_multi = getattr(config_seru, "max_num_of_multiple_task", 0)
        w2task = getattr(excel_loader, "worker_to_task_dict", {}) or {}

        weakness_by_seru: List[Dict[Any, float]] = []
        
        for seru in getattr(best_formation, "seru_set", []):
            workers = getattr(seru, "workers_set", []) or []
            if len(workers) == 0 or len(product_types) == 0:
                weakness_by_seru.append({})
                continue

            cap: Dict[Any, float] = {}
            for t in product_types:
                s_val = 0.0
                for wid in workers:
                    wid = int(wid)
                    real_wid = wid
                    if hasattr(config_seru, 'worker_map') and config_seru.worker_map:
                        real_wid = int(config_seru.worker_map.get(wid, wid))
                    coeff = 0.0
                    if real_wid in w2task and isinstance(w2task[real_wid], dict):
                        coeff = float(w2task[real_wid].get("系数", 0.0) or 0.0)
                    c_i = 1.0 + coeff * (float(N) - float(max_multi))
                    
                    skill = 0.0
                    if real_wid in w2p and isinstance(w2p[real_wid], dict):
                        skill = float(w2p[real_wid].get(t, 0.0) or 0.0)
                    s_val += c_i * skill
                
                cap[t] = s_val / max(len(workers), 1)

            vals = list(cap.values())
            vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 0.0)
            denom = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0

            weakness: Dict[Any, float] = {}
            for t, v in cap.items():
                cap_norm = (v - vmin) / denom
                weakness[t] = 1.0 - cap_norm
            weakness_by_seru.append(weakness)

        self._weakness_by_seru = weakness_by_seru

    def worker_pair_weight_dynamic(self, seru_idx: int, logit_val: float) -> float:
        """
        [优化3] 直接使用预计算的 logit_val，移除 math.pow 和 math.log 计算
        """
        # 1. 获取上下文
        avg_weakness = 0.5
        if 0 <= seru_idx < len(self._weakness_by_seru):
            wk_dict = self._weakness_by_seru[seru_idx]
            if wk_dict:
                avg_weakness = sum(wk_dict.values()) / len(wk_dict)
        
        # 2. 变换计算 (仅做加法和 Sigmoid)
        # z = logit_val + alpha * (avg_weakness - 0.5)
        z = logit_val + self.alpha * (avg_weakness - 0.5)
        w_eff = self._sigmoid(z)

        # 3. 硬约束阈值
        if self.theta_hard_worker is not None and w_eff >= self.theta_hard_worker:
            return 10.0 * w_eff
        return w_eff

    def batch_pair_weight_dynamic(self, seru_idx: int, logit_val: float, type_p: Any, type_q: Any) -> float:
        """
        [优化3] 直接使用预计算的 logit_val
        """
        if 0 <= seru_idx < len(self._weakness_by_seru):
            wk = self._weakness_by_seru[seru_idx]
            wp = float(wk.get(type_p, 0.5)) if wk else 0.5
            wq = float(wk.get(type_q, 0.5)) if wk else 0.5
            pair_weak = 0.5 * (wp + wq)
        else:
            pair_weak = 0.5

        z = logit_val + self.alpha * (pair_weak - 0.5)
        return self._sigmoid(z)

    def compute_r(self, formation: SeruFormation, schedule: SeruSchedule,
                  excel_loader: ExcelDataLoader,
                  worker_mode: str = "base",
                  batch_mode: str = "base",
                  config_seru=None) -> Tuple[float, float, float]:
        """
        计算冲突违背率 r = (P_W + P_B) / W_tot
        修复核心问题：对齐 1-based ID (GA/Excel) 和 0-based Index (Adj Matrix)
        """
        
        # --- 1. 工人冲突 (P_W) ---
        P_W = 0.0
        serus = getattr(formation, "seru_set", []) or []
        worker_seru = {}
        
        # worker_seru 的 key 是 1-based ID (来自 GA chromosome)
        for s_idx, seru in enumerate(serus):
            for wid in getattr(seru, "workers_set", []) or []:
                worker_seru[int(wid)] = s_idx

        # self.worker_adj 是 0-based (i=0 对应 Worker 1)
        for i, nbrs in enumerate(self.worker_adj):
            # [修正] i 是 0-based index -> 转换为 1-based ID
            wid_real_id = i + 1 
            si = worker_seru.get(wid_real_id)
            
            # 如果这个工人没被分配 Seru（异常情况），跳过
            if si is None: continue
            
            # nbrs 中的 j 也是 0-based index
            for j, w, logit_val in nbrs:
                if j <= i: continue 
                
                # [修正] j 是 0-based index -> 转换为 1-based ID
                wjd_real_id = j + 1
                sj = worker_seru.get(wjd_real_id)
                
                # 如果两人在同一个 Seru (si == sj)，则计算冲突
                if sj is not None and si == sj:
                    if worker_mode == "base":
                        P_W += float(w)
                    else:
                        P_W += self.worker_pair_weight_dynamic(si, logit_val)

        # --- 2. 批次冲突 (P_B) ---
        P_B = 0.0
        batches_assignment = getattr(schedule, "batches_assignment", [])
        
        # 将批次分配转为列表结构
        if not batches_assignment:
            seru_batches = [list(getattr(seru, "batches_set", []) or []) for seru in serus]
        else:
            seru_batches = [[] for _ in range(len(serus))]
            for i, group in enumerate(batches_assignment):
                if len(serus) > 0:
                    group_list = list(group) if hasattr(group, '__iter__') else []
                    seru_batches[i % len(serus)].extend(group_list)

        b2p = getattr(excel_loader, "batch_to_product_dict", {}) or {}
        
        for s_idx, blist in enumerate(seru_batches):
            if not blist: continue
            # bset 存储当前 Seru 包含的 1-based Batch ID，用于 O(1) 查找
            bset = set(int(b) for b in blist) 
            
            for p_id in blist: 
                p_id = int(p_id) # 这里拿到的是 1-based ID
                
                # [修正] ID 转 Index，用于访问邻接表
                p_idx = p_id - 1
                
                # 安全检查：防止 ID 越界导致崩溃
                if p_idx < 0 or p_idx >= len(self.batch_adj): 
                    continue
                
                # 遍历 p 的邻居（只遍历预计算好的 Top-K）
                for q_idx, c_pq, logit_val in self.batch_adj[p_idx]:
                    # q_idx 是 0-based index
                    if q_idx <= p_idx: continue 
                    
                    # [修正] Index 转 ID，用于检查是否在当前 Seru (bset) 中
                    q_id = q_idx + 1
                    
                    if q_id in bset:
                        if batch_mode == "base":
                            P_B += float(c_pq)
                        else:
                            # 映射逻辑 ID 到 真实 ID (Excel Key)，用于查询产品类型
                            # 注意：map 和 b2p 通常都使用 1-based ID 逻辑
                            rp, rq = p_id, q_id
                            if config_seru is not None and hasattr(config_seru, 'batch_map') and config_seru.batch_map:
                                rp = int(config_seru.batch_map.get(rp, rp))
                                rq = int(config_seru.batch_map.get(rq, rq))
                            
                            tp = b2p.get(rp, {}).get('产品类型')
                            tq = b2p.get(rq, {}).get('产品类型')
                            
                            w_eff = self.batch_pair_weight_dynamic(s_idx, logit_val, tp, tq)
                            
                            # 硬约束惩罚
                            if self.theta_hard_batch is not None and w_eff >= self.theta_hard_batch:
                                P_B += 10.0 * float(w_eff)
                            else:
                                P_B += float(w_eff)

        r = (P_W + P_B) / self.W_tot
        return float(r), float(P_W), float(P_B)


# ------------------------------
# CCEA 主程序
# ------------------------------
class CCEA:
    def __init__(self, worker_map=None, batch_map=None, edge_scores_worker=None, edge_scores_batch=None, PSF=None, PSS=None):
        
        self.config_seru = ConfigLoader.get_config('config_seru')
        self.config_ccea = ConfigLoader.get_config('config_ccea')

        self.PSF = PSF
        self.PSS = PSS

        # --- 外部映射：逻辑ID -> 真实ID（用于从 Excel 字典读取真实数据）---
        self.worker_map = worker_map or {}
        self.batch_map = batch_map or {}
        self.edge_scores_worker = edge_scores_worker
        self.edge_scores_batch = edge_scores_batch

        if self.worker_map:
            self.config_seru.worker_map = self.worker_map
            # 保持规模一致（如不一致会导致编码长度/矩阵维度错位）
            try:
                self.config_seru.num_of_workers = len(self.worker_map)
            except Exception:
                pass

        if self.batch_map:
            self.config_seru.batch_map = self.batch_map
            try:
                self.config_seru.num_of_batches = len(self.batch_map)
            except Exception:
                pass

        self.loader = ExcelDataLoader.instance()

        # 数据完整性检查
        worker_data = getattr(self.loader, "worker_to_product_dict", None)
        batch_data = getattr(self.loader, "batch_to_product_dict", None)

        if not worker_data or not batch_data:
            print("[CCEA] Data loader not fully initialized. Loading data now...")
            self.loader.read_data(excel_path=self.config_seru.seru_data_path, config_sheet=self.config_seru)
            self.loader.read_data(excel_path=self.config_seru.due_dates_path, config_sheet=self.config_seru)
            self.loader.read_data(excel_path=self.config_seru.batch_types_path, config_sheet=self.config_seru)
        
        W = int(getattr(self.config_seru, "num_of_workers", 0))
        B = int(getattr(self.config_seru, "num_of_batches", 0))
        
        # 1. 处理 Worker Conf
        if self.edge_scores_worker is not None:
            # 先转成标准的 (W, W) 矩阵，范围 [0, 1]
            score_mat_w = self._as_numpy_scores(self.edge_scores_worker, W, "edge_scores_worker")
            # 逻辑反转: Conf = 1 - Score
            w_conf_matrix = 1.0 - score_mat_w
            # 修正: 确保非负，且对角线(自己对自己)冲突为0
            w_conf_matrix = np.clip(w_conf_matrix, 0.0, 1.0)
            np.fill_diagonal(w_conf_matrix, 0.0)
        else:
            # 如果没有提供 scores，回退到全0或者报错
            w_conf_matrix = np.zeros((W, W), dtype=float)

        # 2. 处理 Batch Conf
        if self.edge_scores_batch is not None:
            score_mat_b = self._as_numpy_scores(self.edge_scores_batch, B, "edge_scores_batch")
            b_conf_matrix = 1.0 - score_mat_b
            b_conf_matrix = np.clip(b_conf_matrix, 0.0, 1.0)
            np.fill_diagonal(b_conf_matrix, 0.0)
        else:
            b_conf_matrix = np.zeros((B, B), dtype=float)

        rho = float(getattr(self.config_ccea, "conflict_rho", 1.5))
        eps = float(getattr(self.config_ccea, "conflict_eps", 1e-6))
        alpha = float(getattr(self.config_ccea, "dynamic_conflict_alpha", 1.0))
        thw = getattr(self.config_ccea, "theta_hard_worker", None)
        thb = getattr(self.config_ccea, "theta_hard_batch", None)
        
        self.conflict_model = ConflictModel(
            worker_conf=w_conf_matrix,
            batch_conf=b_conf_matrix,
            rho=rho, eps=eps, alpha=alpha, 
            theta_hard_worker=float(thw) if thw else None,
            theta_hard_batch=float(thb) if thb else None
        )

        self.r_target = float(getattr(self.config_ccea, "conflict_r_target", 0.10))
        self.eta = float(getattr(self.config_ccea, "conflict_eta", 0.1))
        self.gamma = float(getattr(self.config_ccea, "conflict_gamma_init", 1.0))
        self.gamma_min = float(getattr(self.config_ccea, "conflict_gamma_min", 0.0))
        self.gamma_max = float(getattr(self.config_ccea, "conflict_gamma_max", 5.0))
        
        self._warned_evaluator = False


    # =========================================================
    # Hot-start（热启动）：根据 GNN edge_scores 构造一个初始解
    # edge_scores_* 约定为 0-based 的方阵：
    #   edge_scores_worker[i-1][j-1] = score(worker i, worker j)
    #   edge_scores_batch[i-1][j-1]  = score(batch  i, batch  j)
    # 分数越高 => 越适合分到同一个 seru（相似度）
    # =========================================================
    def _as_numpy_scores(self, M, size: int, name: str) -> np.ndarray:
        """把输入 edge_scores 规范成 (size,size) 的 numpy float 矩阵，必要时对称化/裁剪。"""
        if M is None:
            raise ValueError(f"{name} is None")
        
        # dict 形式：{i: {j: score}}
        if isinstance(M, dict):
            A = np.zeros((size, size), dtype=float)
            
            # [修复] 预先检测是否为 0-based
            # 逻辑：只要 keys 里包含 0 或 "0"，就认为是 0-based，不进行 shift
            all_keys = [int(k) for k in M.keys()]
            is_0_based = (0 in all_keys)
            offset = 0 if is_0_based else 1

            for ki, row in M.items():
                # 使用统一的 offset
                i = int(ki) - offset
                
                if i < 0 or i >= size:
                    continue
                    
                if isinstance(row, dict):
                    # 对内层字典也做同样的检测太慢且没必要，通常内外一致
                    # 这里假设内层索引规则与外层一致
                    for kj, v in row.items():
                        j = int(kj) - offset
                        if 0 <= j < size:
                            A[i, j] = float(v)
                else:
                    # {i: [..]} 列表形式，通常隐含 index 0..N
                    try:
                        row_list = list(row)
                        for j in range(min(size, len(row_list))):
                            A[i, j] = float(row_list[j])
                    except Exception:
                        pass
        else:
            # numpy array 或 list of lists
            A = np.array(M, dtype=float)

        if A.ndim != 2:
            raise ValueError(f"{name} must be 2D matrix, got shape={getattr(A,'shape',None)}")

        # 兼容 1-based 矩阵 (多一行一列占位的情况)
        # 只有当维度明确大 1 圈时才裁剪
        if A.shape == (size + 1, size + 1):
            A = A[1:, 1:]
        elif A.shape != (size, size):
            raise ValueError(f"{name} shape mismatch, expect ({size},{size}) or ({size+1},{size+1}), got {A.shape}")

        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(A, 0.0)
        # 对称化
        A = 0.5 * (A + A.T)
        A = np.clip(A, 0.0, 1.0)
        return A

    def _choose_k_serus(self, n_workers: int, n_batches: int) -> int:
        """选择热启动时的 seru 数 K。默认取 round(sqrt(W))，并做边界裁剪。"""
        if n_workers <= 1:
            return 1
        K = int(round(math.sqrt(n_workers)))
        K = max(2, min(K, n_workers))
        if n_batches > 0:
            K = min(K, n_batches)  # seru 数不应超过批次数
        return K

    def _partition_by_scores(self, S: np.ndarray, K: int) -> List[List[int]]:
        """基于相似度矩阵 S (n,n) 做一个快速的“平衡聚类”，输出 1-based id 的簇列表。"""
        n = S.shape[0]
        if K <= 1 or n <= 1:
            return [[i + 1] for i in range(n)]

        # degree 用于种子选择
        S2 = S.copy()
        np.fill_diagonal(S2, -1e9)
        deg = S2.sum(axis=1)

        # kmeans++ 风格选 seed：先选度最高，再选“与已有seed最不相似”的点
        seeds = [int(np.argmax(deg))]
        for _ in range(1, K):
            max_sim = np.max(S[:, seeds], axis=1)  # 与任一 seed 的最大相似度
            dist = 1.0 - max_sim
            dist[seeds] = -1e9
            seeds.append(int(np.argmax(dist)))

        clusters = [[s] for s in seeds]
        sizes = [1 for _ in range(K)]
        # 容量控制（避免极端不平衡）
        cap = int(math.ceil(n / K))

        remaining = [i for i in range(n) if i not in seeds]
        remaining.sort(key=lambda i: float(deg[i]), reverse=True)

        for i in remaining:
            sims = []
            for c in clusters:
                sims.append(float(np.mean(S[i, c])) if c else -1e9)
            order = list(np.argsort(sims))[::-1]
            chosen = None
            for k_idx in order:
                if sizes[int(k_idx)] < cap + 1:
                    chosen = int(k_idx)
                    break
            if chosen is None:
                chosen = int(order[0])
            clusters[chosen].append(i)
            sizes[chosen] += 1

        # 簇内排序：按簇内相似度和从高到低排
        out = []
        for c in clusters:
            if len(c) <= 1:
                ordered = c
            else:
                score = {i: float(np.sum(S[i, c])) for i in c}
                ordered = sorted(c, key=lambda x: score[x], reverse=True)
            out.append([i + 1 for i in ordered])

        # 按簇“代表点”排序，保证稳定性
        out.sort(key=lambda lst: lst[0])
        return out

    def _clusters_to_formation_code(self, clusters: List[List[int]], n_workers: int) -> List[int]:
        """把 worker clusters 转成 formation_code（长度 2W 的 permutation）。"""
        # 去重 + 补全
        used = set()
        norm = []
        for cl in clusters:
            cl2 = []
            for w in cl:
                w = int(w)
                if 1 <= w <= n_workers and w not in used:
                    used.add(w)
                    cl2.append(w)
            if cl2:
                norm.append(cl2)
        missing = [w for w in range(1, n_workers + 1) if w not in used]
        if not norm:
            norm = [missing] if missing else [[1]]
            missing = []
        else:
            if missing:
                norm[-1].extend(missing)

        K = len(norm)
        seps = list(range(n_workers + 1, 2 * n_workers + 1))  # W 个分隔符
        code = []
        sep_i = 0
        for idx, cl in enumerate(norm):
            code.extend(cl)
            if idx < K - 1:
                code.append(seps[sep_i])
                sep_i += 1
        # 剩余分隔符直接放末尾（不会生成空 seru）
        while sep_i < len(seps):
            code.append(seps[sep_i])
            sep_i += 1

        # 安全校验：应为 2W 的 permutation
        if len(code) != 2 * n_workers:
            raise ValueError(f"formation_code length must be 2W={2*n_workers}, got {len(code)}")
        if set(code) != set(range(1, 2 * n_workers + 1)):
            raise ValueError("formation_code is not a permutation of [1..2W]")
        return code

    def _clusters_to_schedule_code(self, clusters: List[List[int]], n_batches: int, n_workers: int) -> List[int]:
        """把 batch clusters 转成 schedule_code（长度 B+W-1 的 permutation）。"""
        # 去重 + 补全
        used = set()
        norm = []
        for cl in clusters:
            cl2 = []
            for b in cl:
                b = int(b)
                if 1 <= b <= n_batches and b not in used:
                    used.add(b)
                    cl2.append(b)
            if cl2:
                norm.append(cl2)
        missing = [b for b in range(1, n_batches + 1) if b not in used]
        if not norm:
            norm = [missing] if missing else [[1]]
            missing = []
        else:
            if missing:
                norm[-1].extend(missing)

        K = len(norm)
        seps = list(range(n_batches + 1, n_batches + n_workers))  # 共 W-1 个分隔符
        code = []
        sep_i = 0
        for idx, cl in enumerate(norm):
            code.extend(cl)
            if idx < K - 1:
                code.append(seps[sep_i])
                sep_i += 1
        # 剩余分隔符放末尾（不会生成空 schedule group）
        while sep_i < len(seps):
            code.append(seps[sep_i])
            sep_i += 1

        # 安全校验：应为 (B+W-1) 的 permutation
        expect_len = n_batches + n_workers - 1
        if len(code) != expect_len:
            raise ValueError(f"schedule_code length must be B+W-1={expect_len}, got {len(code)}")
        if set(code) != set(range(1, n_batches + n_workers)):
            raise ValueError("schedule_code is not a permutation of [1..B+W-1]")
        return code

    def _build_hotstart_solution(self, edge_scores_worker, edge_scores_batch, n_workers: int, n_batches: int) -> Tuple[SeruFormation, SeruSchedule]:
        """根据 edge_scores 构造 (formation, schedule) 作为热启动个体。"""
        K = self._choose_k_serus(n_workers, n_batches)
        Sw = self._as_numpy_scores(edge_scores_worker, n_workers, "edge_scores_worker")
        Sb = self._as_numpy_scores(edge_scores_batch, n_batches, "edge_scores_batch")
        
        worker_clusters = self._partition_by_scores(Sw, K)
        batch_clusters = self._partition_by_scores(Sb, K)

        f_code = self._clusters_to_formation_code(worker_clusters, n_workers)
        s_code = self._clusters_to_schedule_code(batch_clusters, n_batches, n_workers)

        f = SeruFormation(formation_code=f_code)
        Initialization.produce_seru_formation(n_workers, f)

        s = SeruSchedule(schedule_code=s_code)
        Initialization.produce_seru_schedule(n_batches, s)

        return f, s

    def _reset_seru_metrics(self, formation: SeruFormation):
        if not formation or not getattr(formation, "seru_set", None):
            return
        for seru in formation.seru_set:
            seru.throughput_time = 0.0
            seru.processing_time = 0.0
            seru.labour_time = 0.0
            seru.tardiness = 0.0
            seru.fitness = 0.0
            seru.batches_set = []

    def _update_gamma(self, r_avg: float):
        g = self.gamma * math.exp(self.eta * (r_avg - self.r_target))
        self.gamma = max(self.gamma_min, min(self.gamma_max, g))

    def _evaluate_solution(self, solution: Solution, 
                           worker_mode: str, 
                           batch_mode: str, 
                           M_ref: float) -> float:
        """
        评估解，支持传入 worker_mode 和 batch_mode
        """
        self._reset_seru_metrics(solution.formation)
        
        # 1. 计算物理指标
        CalculateFitness.calculate_fitness(solution=solution, config_seru=self.config_seru)
        
        M = float(getattr(solution, "makespan", 0.0) or getattr(solution, "fitness", 0.0))
        
        # 2. 计算冲突率 (传入双模式)
        r, P_W, P_B = self.conflict_model.compute_r(
            formation=solution.formation,
            schedule=solution.schedule,
            excel_loader=self.loader,
            worker_mode=worker_mode,
            batch_mode=batch_mode,
            config_seru=self.config_seru
        )

        # 3. 罚函数计算
        penalized_fitness = M + self.gamma * M_ref * r

        solution.makespan = M 
        solution.fitness = penalized_fitness
        if solution.formation: solution.formation.fitness = penalized_fitness
        if solution.schedule: solution.schedule.fitness = penalized_fitness

        solution._conflict_r = r
        solution._conflict_PW = P_W
        solution._conflict_PB = P_B
        
        return r

    def _tournament_selection(self, population: List[Any]) -> Any:
        a, b = random.sample(population, 2)
        return a if a.fitness <= b.fitness else b
    

    def run(self, Pop_Size, iteration_stop=10000, stop_time = 1000) -> Tuple[SeruFormation, SeruSchedule, Solution]:
        N_W = self.config_seru.num_of_workers
        N_B = self.config_seru.num_of_batches
        pop_size = Pop_Size
        PSF, PSS  = self.PSF, self.PSS

        cmax_his, cmax_his_30, fitness_his = [], [], []

        # --- Hot-start：用 edge_scores 生成一个更“合理”的初始解（覆盖 PSF[0]/PSS[0]） ---
        if pop_size > 0 and self.edge_scores_worker is not None and self.edge_scores_batch is not None:
            try:
                H = int(self.config_ccea.hotstart_frac* pop_size)
                # 保险：至少 1 个，最多 pop_size
                H = max(1, min(H, pop_size))
                f_hot, s_hot = self._build_hotstart_solution(self.edge_scores_worker, self.edge_scores_batch, N_W, N_B)
                PSF[0] = f_hot
                PSS[0] = s_hot

                for idx in range(1, H):
                    f_code = list(f_hot.formation_code)
                    s_code = list(s_hot.schedule_code)

                    # 扰动强度：你可调大/调小（越大越“散”）
                    steps = 3
                    for _ in range(steps):
                        i, j = random.sample(range(len(f_code)), 2)
                        f_code[i], f_code[j] = f_code[j], f_code[i]
                    for _ in range(steps):
                        i, j = random.sample(range(len(s_code)), 2)
                        s_code[i], s_code[j] = s_code[j], s_code[i]

                    f = SeruFormation(formation_code=f_code)
                    Initialization.produce_seru_formation(N_W, f)
                    PSF[idx] = f

                    s = SeruSchedule(schedule_code=s_code)
                    Initialization.produce_seru_schedule(N_B, s)
                    PSS[idx] = s
                print("[HotStart] Seeded initial formation/schedule from edge_scores.")
            except Exception as e:
                print(f"[HotStart] skipped due to error: {e}")

        best_formation = PSF[0]
        best_schedule = PSS[0]
        self.conflict_model.update_dynamic_by_formation(best_formation, self.config_seru, self.loader)
        
        curr_sol = Solution(formation=best_formation, schedule=best_schedule)
        # 初始评估：两边都开 Dynamic，确保起点准确
        self._evaluate_solution(curr_sol, worker_mode="dynamic", batch_mode="dynamic", M_ref=1.0)
        M_ref = float(curr_sol.makespan if curr_sol.makespan > 1e-5 else 1.0)
        
        self._evaluate_solution(curr_sol, worker_mode="dynamic", batch_mode="dynamic", M_ref=M_ref)
        best_solution = curr_sol
        curr_best_cmax = copy.deepcopy(curr_sol)

        start_time = time.time()
        iteration = 0
        iteration_stop = iteration_stop
        iteration_last = 0
        cmax_his.append([best_solution.makespan,0])
        fitness_his.append(best_solution.fitness)
        print(f"Start CCEA. Init MS: {best_solution.makespan:.2f}, r: {best_solution._conflict_r:.4f}")

        while (iteration-iteration_last < iteration_stop and time.time() - start_time <= stop_time):
            iteration += 1
            M_ref = float(best_solution.makespan if best_solution.makespan > 1e-5 else 1.0)

            # =========================================================
            # Stage 1: Evolve Formation (进化构造)
            # 策略：动态构造冲突 (Dynamic Worker) + 静态调度冲突 (Static Batch)
            # =========================================================
            r_vals = []
            for f in PSF:
                sol = Solution(formation=f, schedule=best_schedule)
                # [模式切换] Worker: Dynamic, Batch: Base
                r = self._evaluate_solution(sol, worker_mode="dynamic", batch_mode="base", M_ref=M_ref)
                r_vals.append(r)
            if r_vals:
                self._update_gamma(sum(r_vals)/len(r_vals))

            new_PSF = []
            selected = [self._tournament_selection(PSF) for _ in range(len(PSF))]
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i+1]
                
                c1 = SeruFormation(formation_code=list(p1.formation_code))
                c2 = SeruFormation(formation_code=list(p2.formation_code))
                
                c1.formation_code, c2.formation_code = GaOperator.order_crossover(
                    p1.formation_code, p2.formation_code, self.config_ccea
                )
                c1.formation_code = GaOperator.swap_mutation(c1.formation_code, self.config_ccea)
                c2.formation_code = GaOperator.swap_mutation(c2.formation_code, self.config_ccea)
                
                Initialization.produce_seru_formation(N_W, c1)
                Initialization.produce_seru_formation(N_W, c2)
                new_PSF.extend([c1, c2])
            PSF = new_PSF

            for f_ind in PSF:
                sol_temp = Solution(formation=f_ind, schedule=best_schedule)
                # 计算 fitness 并回写到 f_ind.fitness
                self._evaluate_solution(sol_temp, worker_mode="dynamic", batch_mode="base", M_ref=M_ref)
                f_ind.fitness = sol_temp.fitness
                f_ind.makespan = sol_temp.makespan

            # 局部搜索 (Formation)
            curr_best_f = min(PSF, key=lambda x: x.fitness)
            def f_eval(sol): 
                # [模式切换] Worker: Dynamic, Batch: Base
                self._evaluate_solution(sol, worker_mode="dynamic", batch_mode="base", M_ref=M_ref)
                return sol.fitness            
            try:
                curr_best_f = GaOperator.local_search_formation(
                    formation=curr_best_f,
                    best_scheduling=best_schedule, 
                    config_seru=self.config_seru,
                    config_ccea=self.config_ccea,
                    evaluator=f_eval
                )
            except TypeError:
                if not self._warned_evaluator:
                    print("[Warning] GaOperator.local_search_formation does not support 'evaluator'. LS skipped.")
                    self._warned_evaluator = True
            
            self.conflict_model.update_dynamic_by_formation(curr_best_f, self.config_seru, self.loader)

            # =========================================================
            # Stage 2: Evolve Schedule (进化调度)
            # 策略：静态构造冲突 (Static Worker) + 动态调度冲突 (Dynamic Batch)
            # =========================================================
            r_vals = []
            for s in PSS:
                sol = Solution(formation=curr_best_f, schedule=s)
                # [模式切换] Worker: Base, Batch: Dynamic
                r = self._evaluate_solution(sol, worker_mode="base", batch_mode="dynamic", M_ref=M_ref)
                r_vals.append(r)
            if r_vals:
                self._update_gamma(sum(r_vals)/len(r_vals))

            new_PSS = []
            selected = [self._tournament_selection(PSS) for _ in range(len(PSS))]
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i+1]
                
                c1 = SeruSchedule(schedule_code=list(p1.schedule_code))
                c2 = SeruSchedule(schedule_code=list(p2.schedule_code))
                
                c1.schedule_code, c2.schedule_code = GaOperator.order_crossover(
                    p1.schedule_code, p2.schedule_code, self.config_ccea
                )
                c1.schedule_code = GaOperator.swap_mutation(c1.schedule_code, self.config_ccea)
                c2.schedule_code = GaOperator.swap_mutation(c2.schedule_code, self.config_ccea)
                
                Initialization.produce_seru_schedule(N_B, c1)
                Initialization.produce_seru_schedule(N_B, c2)
                new_PSS.extend([c1, c2])
            PSS = new_PSS
            
            for s_ind in PSS:
                sol_temp = Solution(formation=curr_best_f, schedule=s_ind) # 注意这里用的是 curr_best_f
                self._evaluate_solution(sol_temp, worker_mode="base", batch_mode="dynamic", M_ref=M_ref)
                s_ind.fitness = sol_temp.fitness

            # 局部搜索 (Schedule)
            curr_best_s = min(PSS, key=lambda x: x.fitness)
            def s_eval(sol):
                # [模式切换] Worker: Base, Batch: Dynamic
                self._evaluate_solution(sol, worker_mode="base", batch_mode="dynamic", M_ref=M_ref)
                return sol.fitness

            try:
                curr_best_s = GaOperator.local_search_schedule(
                    schedule=curr_best_s,
                    best_formation=curr_best_f,
                    config_seru=self.config_seru,
                    config_ccea=self.config_ccea,
                    evaluator=s_eval
                )
            except TypeError:
                pass 

            # ==========================
            # 更新全局最优
            # ==========================
            curr_sol = Solution(formation=curr_best_f, schedule=curr_best_s)
            # 全局评估时使用双 Dynamic 确保精确
            self._evaluate_solution(curr_sol, worker_mode="dynamic", batch_mode="dynamic", M_ref=M_ref)
            
            if curr_sol.makespan < best_solution.makespan:
                best_solution = curr_sol
                iteration_last = iteration
                
            curr_best_cmax1 = min(PSS, key=lambda x: x.makespan)
            curr_best_cmax2 = min(PSF, key=lambda x: x.makespan)
            # curr_best_cmax3 = min(curr_best_cmax1, curr_best_cmax2, key=lambda x: x.makespan)
            curr_best_cmax = min(curr_best_cmax, curr_best_cmax1, curr_best_cmax2, key=lambda x: x.makespan)
            elapsed = time.time() - start_time
            cmax_his.append([best_solution.makespan, elapsed])
            fitness_his.append(best_solution.fitness)
            if len(cmax_his_30) == 0 and elapsed >= 30:
                cmax_his_30 = cmax_his.copy()
            if iteration % 10 == 0:
                
                
                print(f"Iter {iteration} | Fit: {best_solution.fitness:.2f} (MS: {best_solution.makespan:.2f}) | min(cmax): {curr_best_cmax.makespan:.2f}  | "
                      f"r: {getattr(best_solution,'_conflict_r',0):.2%} | gamma: {self.gamma:.2f} | t: {elapsed:.1f}s")

        return best_solution.formation, best_solution.schedule, best_solution, cmax_his, cmax_his_30, fitness_his, iteration


def main():
    ConfigLoader.preload_all()
    ccea = CCEA()
    bf, bs, bsol = ccea.run()

    print("\n=== Final Results ===")
    print(f"Makespan: {bsol.makespan}")
    print(f"Conflict r: {getattr(bsol, '_conflict_r', 0.0):.4f} (PW: {getattr(bsol, '_conflict_PW', 0):.2f}, PB: {getattr(bsol, '_conflict_PB', 0):.2f})")
    print(f"Fitness: {bsol.fitness:.4f}")

if __name__ == "__main__":
    main()