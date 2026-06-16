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
    ):
        self.rho = float(rho)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.theta_hard_worker = theta_hard_worker
        self.theta_hard_batch = theta_hard_batch

        self.worker_conf = self._symmetrize_and_clip(worker_conf)
        self.batch_conf = self._symmetrize_and_clip(batch_conf)

        # 邻接表（稀疏化）
        self.worker_adj = self._build_adj(self.worker_conf, eps=self.eps)
        self.batch_adj = self._build_adj(self.batch_conf, eps=self.eps)

        # W_tot 计算（分母）
        self.W_tot = self._total_weight(self.worker_adj) + self._total_weight(self.batch_adj)
        if self.W_tot <= 1e-9:
            self.W_tot = 1.0  

        # 动态条件化缓存
        self._weakness_by_seru: List[Dict[Any, float]] = []
    def _calculate_single_seru_avg_weakness(self, worker_ids: List[int], excel_loader, config_seru) -> float:
        """[新增] 实时计算单个 Seru 的平均 Weakness，不依赖全局缓存"""
        if not worker_ids:
            return 0.5
            
        w2p = getattr(excel_loader, "worker_to_product_dict", {}) or {}
        if not w2p: return 0.5

        # 1. 收集所有产品类型
        product_types = set()
        # 这里为了效率，可以考虑把 product_types 缓存到类成员，不要每次都遍历
        # 但为保证逻辑独立性，暂时保留或简化
        if hasattr(self, '_cached_product_types') and self._cached_product_types:
             product_types = self._cached_product_types
        else:
             for mp in w2p.values():
                 for t in mp.keys():
                     product_types.add(t)
             self._cached_product_types = sorted(list(product_types))
             product_types = self._cached_product_types

        if not product_types: return 0.5

        N = getattr(config_seru, "num_of_workers", 0)
        max_multi = getattr(config_seru, "max_num_of_multiple_task", 0)
        w2task = getattr(excel_loader, "worker_to_task_dict", {}) or {}

        # 2. 计算 Capability
        cap_vals = []
        for t in product_types:
            s_val = 0.0
            for wid in worker_ids:
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
            
            cap_vals.append(s_val / max(len(worker_ids), 1))

        # 3. 归一化并转为 Weakness
        vmin, vmax = min(cap_vals), max(cap_vals)
        denom = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
        
        weaknesses = [(1.0 - (v - vmin) / denom) for v in cap_vals]
        return sum(weaknesses) / len(weaknesses)
        
    @staticmethod
    def _symmetrize_and_clip(M: np.ndarray) -> np.ndarray:
        M = np.array(M, dtype=float)
        M = np.clip(M, 0.0, 1.0)
        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            M = 0.5 * (M + M.T)
            np.fill_diagonal(M, 0.0)
        return M

    @staticmethod
    def _build_adj(M: np.ndarray, eps: float) -> List[List[Tuple[int, float]]]:
        n = M.shape[0]
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            for j, w in enumerate(M[i]):
                if j != i and w > eps:
                    adj[i].append((j, float(w)))
        return adj

    @staticmethod
    def _total_weight(adj: List[List[Tuple[int, float]]]) -> float:
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

    def update_dynamic_by_formation(self, best_formation: SeruFormation, config_seru, excel_loader: ExcelDataLoader):
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
                    # 使用 worker_map: 逻辑工人ID -> 真实工人ID (Excel 字典键)
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

    def worker_pair_weight_dynamic(self, seru_idx: int, w_raw: float) -> float:
        """
        [修改] 计算工人的动态冲突权重 - 与批次计算逻辑完全统一
        逻辑：
        1. 获取当前 Seru 的平均 Weakness（代表环境恶劣程度）。
        2. 应用 Logit + Alpha偏移 + Sigmoid 变换。
        3. 应用硬约束阈值惩罚。
        """
        if w_raw <= self.eps:
            return 0.0
            
        # 1. 获取上下文：Seru 的平均 Weakness
        # 如果 Seru 很弱 (Weakness高)，环境压力大，工人冲突权重放大
        avg_weakness = 0.5
        if 0 <= seru_idx < len(self._weakness_by_seru):
            wk_dict = self._weakness_by_seru[seru_idx]
            if wk_dict:
                # 计算该 Seru 对所有产品类型的平均弱点
                avg_weakness = sum(wk_dict.values()) / len(wk_dict)
        
        # 2. 变换计算 (与 batch_pair_weight_dynamic 保持一致)
        base = float(w_raw) ** self.rho
        z = self._logit(base, eps=self.eps)
        # 核心逻辑：(avg_weakness - 0.5) > 0 (弱) -> z 变大 -> w_eff 变大
        z = z + self.alpha * (avg_weakness - 0.5)
        w_eff = self._sigmoid(z)

        # 3. 硬约束阈值
        if self.theta_hard_worker is not None and w_eff >= self.theta_hard_worker:
            return 10.0 * w_eff
        return w_eff

    def batch_pair_weight_dynamic(self, seru_idx: int, base_conf: float, type_p: Any, type_q: Any) -> float:
        """根据 Seru 的 weakness 动态调整批次对冲突权重"""
        if base_conf <= self.eps:
            return 0.0
        
        if 0 <= seru_idx < len(self._weakness_by_seru):
            wk = self._weakness_by_seru[seru_idx]
            wp = float(wk.get(type_p, 0.5)) if wk else 0.5
            wq = float(wk.get(type_q, 0.5)) if wk else 0.5
            pair_weak = 0.5 * (wp + wq)
        else:
            pair_weak = 0.5

        base = float(base_conf) ** self.rho
        z = self._logit(base, eps=self.eps)
        z = z + self.alpha * (pair_weak - 0.5)
        return self._sigmoid(z)

    def compute_r(self, formation: SeruFormation, schedule: SeruSchedule,
                excel_loader: ExcelDataLoader,
                worker_mode: str = "base",
                batch_mode: str = "base",
                config_seru=None) -> Tuple[float, float, float]:
        """
        计算总违背率 r，支持双模态（Static/Dynamic）切换。
        
        修正点：
        1. 1-based ID (GA) 转 0-based Index (Matrix)。
        2. [Worker Dynamic] 模式下，实时计算 Seru Weakness，修复代际滞后问题。
        """

        serus = getattr(formation, "seru_set", []) or []

        # --- 1. 工人冲突 (P_W): seru 内部 pair ---
        P_W = 0.0
        Wn = int(getattr(self.worker_conf, "shape", [0])[0])
        
        for s_idx, seru in enumerate(serus):
            wlist = getattr(seru, "workers_set", []) or []
            if not wlist:
                continue
            wuniq = sorted({int(w) for w in wlist})
            L = len(wuniq)
            if L <= 1:
                continue

            # [关键修复] 实时计算当前 Seru 的 Weakness
            # 如果是 dynamic 模式（进化构造阶段），不能查缓存，必须算实时的
            current_seru_weakness = 0.5
            if worker_mode == "dynamic":
                 current_seru_weakness = self._calculate_single_seru_avg_weakness(wuniq, excel_loader, config_seru)

            for a in range(L - 1):
                wid_i = wuniq[a]
                if wid_i <= 0 or wid_i > Wn: continue # 越界保护
                i0 = wid_i - 1 # 1-based -> 0-based
                
                row = self.worker_conf[i0]
                
                for b in range(a + 1, L):
                    wid_j = wuniq[b]
                    if wid_j <= 0 or wid_j > Wn: continue
                    j0 = wid_j - 1
                    
                    w_raw = float(row[j0])
                    if w_raw <= self.eps: continue # 无冲突，跳过

                    if worker_mode == "base":
                        P_W += w_raw
                    else:
                        # [关键修复] 使用实时 weakness 进行内联计算
                        # 公式: Sigmoid( Logit(w^rho) + alpha * (Weakness - 0.5) )
                        base = w_raw ** self.rho
                        z = self._logit(base, eps=self.eps)
                        z = z + self.alpha * (current_seru_weakness - 0.5)
                        w_eff = self._sigmoid(z)
                        
                        # 硬约束惩罚
                        if self.theta_hard_worker is not None and w_eff >= self.theta_hard_worker:
                            P_W += 10.0 * w_eff
                        else:
                            P_W += w_eff

        # --- 2. 批次冲突 (P_B): seru 内部 pair ---
        P_B = 0.0
        batches_assignment = getattr(schedule, "batches_assignment", [])

        if not batches_assignment:
            seru_batches = [list(getattr(seru, "batches_set", []) or []) for seru in serus]
        else:
            seru_batches = [[] for _ in range(len(serus))]
            for i, group in enumerate(batches_assignment):
                if len(serus) > 0:
                    group_list = list(group) if hasattr(group, '__iter__') else []
                    seru_batches[i % len(serus)].extend(group_list)

        b2p = getattr(excel_loader, "batch_to_product_dict", {}) or {}

        # 轻量缓存：logical batch id -> product type
        type_cache: Dict[int, Any] = {}

        def _get_type(logical_bid: int) -> Any:
            if logical_bid in type_cache:
                return type_cache[logical_bid]
            real_bid = logical_bid
            if config_seru is not None and hasattr(config_seru, 'batch_map') and getattr(config_seru, "batch_map", None):
                real_bid = int(config_seru.batch_map.get(int(logical_bid), int(logical_bid)))
            t = b2p.get(int(real_bid), {}).get('产品类型')
            type_cache[logical_bid] = t
            return t

        Bn = int(getattr(self.batch_conf, "shape", [0])[0])
        for s_idx, blist in enumerate(seru_batches):
            if not blist:
                continue
            buniq = sorted({int(b) for b in blist})
            L = len(buniq)
            if L <= 1:
                continue
            
            # 批次部分不需要实时计算 Weakness
            # 因为在 Stage 2 (进化调度) 时，Formation 是固定的，Seru 环境也是固定的。
            # 直接使用 cached weakness 即可。
            
            for a in range(L - 1):
                bid_p = buniq[a]
                if bid_p <= 0 or bid_p > Bn: continue
                p0 = bid_p - 1 # 1-based -> 0-based
                
                row = self.batch_conf[p0]
                
                for b in range(a + 1, L):
                    bid_q = buniq[b]
                    if bid_q <= 0 or bid_q > Bn: continue
                    q0 = bid_q - 1
                    
                    c_pq = float(row[q0])
                    if c_pq <= self.eps: continue

                    if batch_mode == "base":
                        P_B += c_pq
                    else:
                        tp = _get_type(bid_p)
                        tq = _get_type(bid_q)
                        # 这里继续调用原来的 batch_pair_weight_dynamic，因为它会查 _weakness_by_seru 缓存
                        w_eff = float(self.batch_pair_weight_dynamic(s_idx, c_pq, tp, tq))
                        
                        if self.theta_hard_batch is not None and w_eff >= self.theta_hard_batch:
                            P_B += 10.0 * w_eff
                        else:
                            P_B += w_eff

        r = (P_W + P_B) / float(self.W_tot)
        return float(r), float(P_W), float(P_B)



def _load_conf_matrix(path: Optional[str], size: int, one_based: bool = True) -> np.ndarray:
    if not path:
        n = size + 1 if one_based else size
        return np.zeros((n, n), dtype=float)
    
    try:
        if "::" in path:
            file_path, sheet = path.split("::", 1)
            df = pd.read_excel(file_path, sheet_name=sheet, header=None)
            M = df.values
        elif path.endswith(".npy"):
            M = np.load(path)
        elif path.endswith(".csv"):
            M = np.loadtxt(path, delimiter=",")
        else:
            df = pd.read_excel(path, header=None)
            M = df.values
    except Exception as e:
        print(f"[Warning] Failed to load matrix {path}: {e}. Using zeros.")
        n = size + 1 if one_based else size
        return np.zeros((n, n), dtype=float)

    M = np.array(M, dtype=float)
    n = size + 1 if one_based else size
    if M.shape != (n, n):
        M2 = np.zeros((n, n), dtype=float)
        r_dim, c_dim = min(n, M.shape[0]), min(n, M.shape[1])
        M2[:r_dim, :c_dim] = M[:r_dim, :c_dim]
        M = M2
    return M


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
            offset = 0 if is_0_based else 1  # 统一偏移量

            for ki, row in M.items():
                i = int(ki) - offset # 使用统一偏移
                
                if i < 0 or i >= size:
                    continue
                if isinstance(row, dict):
                    for kj, v in row.items():
                        j = int(kj) - offset
                        if 0 <= j < size:
                            A[i, j] = float(v)
                else:
                    try:
                        row_list = list(row)
                        for j in range(min(size, len(row_list))):
                            A[i, j] = float(row_list[j])
                    except Exception:
                        pass
        else:
            A = np.array(M, dtype=float)

        if A.ndim != 2:
            raise ValueError(f"{name} must be 2D matrix, got shape={getattr(A,'shape',None)}")

        if A.shape == (size + 1, size + 1):
            A = A[1:, 1:]
        elif A.shape != (size, size):
            raise ValueError(f"{name} shape mismatch, expect ({size},{size}) or ({size+1},{size+1}), got {A.shape}")

        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(A, 0.0)
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

        # 注意：formation/schedule 的 makespan 在本实现中用于“给定另一阶段固定时”的联合 makespan（便于 min(cmax) 选择）
        if solution.formation is not None:
            solution.formation.fitness = penalized_fitness
            solution.formation.makespan = M
        if solution.schedule is not None:
            solution.schedule.fitness = penalized_fitness
            solution.schedule.makespan = M

        solution._conflict_r = r
        solution._conflict_PW = P_W
        solution._conflict_PB = P_B

        return r


    def _tournament_selection(self, population: List[Any]) -> Any:
        a, b = random.sample(population, 2)
        return a if a.fitness <= b.fitness else b
    

    def run(self, Pop_Size: Optional[int] = None) -> Tuple[SeruFormation, SeruSchedule, Solution]:
        N_W = self.config_seru.num_of_workers
        N_B = self.config_seru.num_of_batches
        # --- pop_size 兼容：支持不传参时从现有 PSF/PSS 或 config 读取 ---
        if Pop_Size is None:
            if isinstance(self.PSF, list) and len(self.PSF) > 0:
                pop_size = len(self.PSF)
            else:
                pop_size = int(getattr(self.config_ccea, "pop_size", 0) or getattr(self.config_ccea, "population_size", 0) or 500)
        else:
            pop_size = int(Pop_Size)
        # --- 若未提供 PSF/PSS，则生成随机初始种群 ---
        if (self.PSF is None) or (self.PSS is None) or (not isinstance(self.PSF, list)) or (not isinstance(self.PSS, list)) or (len(self.PSF) < pop_size) or (len(self.PSS) < pop_size):
            PSF = []
            PSS = []

            def _random_formation_code(n_workers: int) -> List[int]:
                workers = list(range(1, n_workers + 1))
                random.shuffle(workers)
                seps = list(range(n_workers + 1, 2 * n_workers + 1))  # W 个分隔符

                # 选择 seru 个数 K（1..min(5,W)），插入 K-1 个分隔符在 workers 之间，其余分隔符放末尾
                Kmax = int(getattr(self.config_ccea, "init_k_max", 5) or 5)
                K = random.randint(1, max(1, min(Kmax, n_workers)))
                cuts = []
                if n_workers > 1 and K > 1:
                    slots = list(range(1, n_workers))
                    random.shuffle(slots)
                    cuts = sorted(slots[:K - 1])

                code = []
                sep_i = 0
                prev = 0
                for cut in cuts + [n_workers]:
                    code.extend(workers[prev:cut])
                    if cut != n_workers:
                        code.append(seps[sep_i]); sep_i += 1
                    prev = cut
                while sep_i < len(seps):
                    code.append(seps[sep_i]); sep_i += 1
                return code

            def _random_schedule_code(n_batches: int, n_workers: int) -> List[int]:
                batches = list(range(1, n_batches + 1))
                random.shuffle(batches)
                seps = list(range(n_batches + 1, n_batches + n_workers))  # 共 W-1 个分隔符

                Kmax = int(getattr(self.config_ccea, "init_k_max", 5) or 5)
                K = random.randint(1, max(1, min(Kmax, n_workers)))
                cuts = []
                if n_batches > 1 and K > 1:
                    slots = list(range(1, n_batches))
                    random.shuffle(slots)
                    cuts = sorted(slots[:min(K - 1, len(slots))])

                code = []
                sep_i = 0
                prev = 0
                for cut in cuts + [n_batches]:
                    code.extend(batches[prev:cut])
                    if cut != n_batches and sep_i < len(seps):
                        code.append(seps[sep_i]); sep_i += 1
                    prev = cut
                while sep_i < len(seps):
                    code.append(seps[sep_i]); sep_i += 1
                return code

            for _ in range(pop_size):
                f_code = _random_formation_code(N_W)
                f = SeruFormation(formation_code=f_code)
                Initialization.produce_seru_formation(N_W, f)
                PSF.append(f)

                s_code = _random_schedule_code(N_B, N_W)
                s = SeruSchedule(schedule_code=s_code)
                Initialization.produce_seru_schedule(N_B, s)
                PSS.append(s)

            self.PSF, self.PSS = PSF, PSS
        else:
            PSF, PSS = self.PSF[:pop_size], self.PSS[:pop_size]
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
        cmax_his.append([best_solution.makespan,0])
        fitness_his.append(best_solution.fitness)
        print(f"Start CCEA. Init MS: {best_solution.makespan:.2f}, r: {best_solution._conflict_r:.4f}")

        max_runtime = float(getattr(self.config_ccea, "max_runtime", float("inf")) or float("inf"))
        max_iter = int(getattr(self.config_ccea, "max_iteration", 0) or getattr(self.config_ccea, "max_iter", 0) or 0)
        while (time.time() - start_time < max_runtime):
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
                s_ind.makespan = sol_temp.makespan

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