from problem.pure_seru.pure_seru_entities import SeruFormation
import numpy as np
from typing import List
from sklearn.metrics import adjusted_rand_score

def _formation_signature(formation, num_workers: int) -> List[int]:
    """
    把一个 SeruFormation 映射成长度 = num_workers 的标签向量。
    下标 = worker ID；值 = 该工人所在 Seru 的编号（先按 Seru 工人数降序再编号）。
    """
    # 1) 先按 |workers_set| 降序排序，确保编号顺序与题意一致
    seru_sorted = sorted(
        formation.seru_set,
        key=lambda s: len(s.workers_set),
        reverse=True
    )
    # 2) 建立 [worker_id] -> seru_idx 的映射
    label_vec = [-1] * num_workers            # -1 表示占位
    # print(seru_sorted.__len__())
    # print("------")
    for seru_idx, seru in enumerate(seru_sorted):
        for w in seru.workers_set:
            label_vec[w-1] = seru_idx
        # print(seru_idx)
    return label_vec

from typing import List

def _formation_signature_sequential(formation) -> List[int]:
    """
    顺序式 signature：
        - 先按 Seru 工人数从大到小排序；
        - 按排序后的顺序依次把 Seru 编号写入 label_vec，中间不跳号。

    例：seru_sorted 的工人数依次为 [3, 2, 1]，则
        label_vec = [0, 0, 0, 1, 1, 2]
    """
    # 1) 先把 seru_set 按工人数降序排好
    seru_sorted = sorted(
        formation.seru_set,
        key=lambda s: len(s.workers_set),
        reverse=True
    )

    # 2) 顺序平铺：重复 seru_idx 次数 = 该 Seru 的工人数
    label_vec: List[int] = []
    for seru_idx, seru in enumerate(seru_sorted):
        label_vec.extend([seru_idx] * len(seru.workers_set))
    # print(label_vec)
    return label_vec


def calculate_ari(formations: List['SeruFormation']) -> List[float]:
    """
    给定多个 SeruFormation，计算：
        对第 i 个 formation，与其余 formation 的 ARI 均值
    返回 [ari_mean_0, ari_mean_1, ...]，长度 = len(formations)
    """
    if not formations:
        return []

    # 1. 每个 formation 转成标签向量
    signatures = [
        # _formation_signature(f, num_workers)
        _formation_signature_sequential(f)
        for f in formations
    ]

    # 2. 计算 pair‑wise ARI，然后求均值
    n = len(signatures)
    ari_means = []
    for i in range(n):
        aris = [
            adjusted_rand_score(signatures[i], signatures[j])
            for j in range(n) if j != i
        ]
        ari_means.append(float(np.mean(aris)))   # 输出普通 float
    return ari_means