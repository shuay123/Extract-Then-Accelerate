# -*- coding: utf-8 -*-
"""
同名类/同接口适配版（Drop-in Adapter）

目标：在不改你现有调用方式的前提下，把原 metaheuristics/ccea/ccea.py 的 CCEA 替换为
Random Keys（优先级染色体）+ 图着色/贪心解码器 + 动态调度冲突更新 + γ 自适应软约束 的分支实现。

你原来的调用方式通常是：
    ConfigLoader.preload_all()
    ccea = CCEA()
    best_formation, best_scheduling, best_solution = ccea.run()

本文件提供同名类 CCEA、同名方法 run()，返回类型不变：
    (SeruFormation, SeruSchedule, Solution)

依赖：请确保以下 4 个分支文件已放入你的项目同目录：
- metaheuristics/ccea/ccea_rk_branch.py
- metaheuristics/ccea/rk_decoders.py
- metaheuristics/ccea/rk_operators.py
- metaheuristics/ccea/conflict_dynamic.py

建议用法：
1) 将本文件拷贝到你项目的 metaheuristics/ccea/ccea.py（覆盖原文件）
   或者将原 ccea.py 改名备份，再把本文件改名为 ccea.py。
2) 按需在 config_seru/config_ccea 添加矩阵路径与超参（见文档）。

备注：
- 默认返回综合最优 elite_fit（makespan + 软约束惩罚）。
- 如果你希望“只看 makespan 最小”的结果，可在 config_ccea 增加：
      return_policy: "elite_ms"
  或设置环境变量：CCEA_RETURN_POLICY=elite_ms
"""

from __future__ import annotations
import os
from typing import Tuple
import os
import sys
# 添加项目根目录到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from utils.config_loader import ConfigLoader
from problem.pure_seru.pure_seru_entities import SeruFormation, SeruSchedule, Solution

# 分支实现（Random Keys + 解码器 + 动态调度冲突）
from metaheuristics.ccea.ccea_rk_branch import CCEA_RK


class CCEA:
    """
    同名类 CCEA：对外接口保持不变。
    内部用 CCEA_RK 执行新分支逻辑。
    """
    def __init__(self):
        # 原项目通常在 main() 里 preload_all；这里做一次兜底，避免用户忘记调用
        try:
            ConfigLoader.preload_all()
        except Exception:
            pass

        self._impl = CCEA_RK()

        # 返回策略：elite_fit（默认）/ elite_ms（只看 makespan）
        self._return_policy = (
            os.getenv("CCEA_RETURN_POLICY")
            or getattr(self._impl.config_ccea, "return_policy", None)
            or "elite_fit"
        ).strip()

    def run(self) -> Tuple[SeruFormation, SeruSchedule, Solution]:
        """
        与原 CCEA.run 同接口：返回 (best_formation, best_schedule, best_solution)。
        - best_solution.fitness：综合适应值（包含软约束罚项）
        - best_solution.makespan：真实 makespan（由 CalculateFitness 计算）
        """
        best_formation, best_schedule, best_solution = self._impl.run()

        # 若用户只要 makespan 最优，从实现内部的 elite_ms 取
        if self._return_policy.lower() in ("elite_ms", "makespan", "ms"):
            elite_ms = getattr(self._impl, "elite_ms", None)
            if elite_ms is not None:
                sol = Solution(formation=elite_ms.formation, schedule=elite_ms.schedule)
                sol.fitness = elite_ms.fitness
                sol.makespan = elite_ms.makespan
                return elite_ms.formation, elite_ms.schedule, sol

        return best_formation, best_schedule, best_solution


def main():
    # 与你原来 ccea.py 的 main 保持一致的使用方式
    ConfigLoader.preload_all()
    ccea = CCEA()
    best_formation, best_scheduling, best_solution = ccea.run()

    print("Best Formation:", best_formation)
    if hasattr(best_formation, "seru_set"):
        for i, seru in enumerate(best_formation.seru_set, start=1):
            ws = getattr(seru, "workers_set", []) or []
            print(f"Seru {i}: {len(ws)} workers")

    print("Best Scheduling:", best_scheduling)
    ba = getattr(best_scheduling, "batches_assignment", []) or []
    for i, g in enumerate(ba, start=1):
        print(f"Seru {i} batches:", g)

    print("Best Solution:", best_solution)


if __name__ == "__main__":
    main()
