# -*- coding: utf-8 -*-
"""
rk_operators.py
Random Keys（连续优先级染色体）上的 GA 算子：交叉 + 变异 + 选择
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


def blend_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """算术混合交叉：child = αa + (1-α)b，α∼U(0,1)（逐位）"""
    alpha = rng.random(size=a.shape)
    c1 = alpha * a + (1.0 - alpha) * b
    c2 = alpha * b + (1.0 - alpha) * a
    return c1, c2


def uniform_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """按位从父代 A/B 随机继承"""
    mask = rng.random(size=a.shape) < 0.5
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1, c2


def gaussian_mutation(x: np.ndarray, sigma: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """高斯扰动：以概率 p 对每位加 N(0,sigma^2)"""
    y = x.copy()
    mask = rng.random(size=y.shape) < p
    y[mask] = y[mask] + rng.normal(loc=0.0, scale=sigma, size=mask.sum())
    # keys 不要求在 [0,1]，但截断能避免数值过大
    return np.clip(y, -5.0, 5.0)


def reset_mutation(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """随机重置：以概率 p 把基因位重置到 U(0,1)"""
    y = x.copy()
    mask = rng.random(size=y.shape) < p
    y[mask] = rng.random(size=mask.sum())
    return y


def tournament_select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """返回被选个体索引（锦标赛规模 k）"""
    idx = rng.integers(0, len(fitness), size=k)
    best = idx[np.argmin(fitness[idx])]
    return int(best)
