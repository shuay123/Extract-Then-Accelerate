import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


ArrayLike2D = Union[np.ndarray, Sequence[Sequence[float]]]


def _as_2d_float(x: ArrayLike2D, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维矩阵，但 got {name}.ndim={arr.ndim}")
    return arr


def plot_multi_matrices(
    matrices: Union[Sequence[ArrayLike2D], Dict[str, ArrayLike2D]],
    labels: Optional[Sequence[str]] = None,
    *,
    save_dir: str = ".",
    prefix: str = "compare",
    dpi: int = 200,
    save_pdf: bool = False,
    x_label: str = "iteration",
    y_label: str = "value",
    per_row_title: Optional[str] = None,
    mean_title: Optional[str] = None,
    ci: Optional[str] = "stderr",  # None | "std" | "stderr"
    ci_alpha: float = 0.18,
    ddof: int = 1,
    nan_policy: str = "propagate",  # "propagate" | "omit"
    show_subplot_legend: bool = False,
    global_legend_loc: str = "upper right",
    max_cols: Optional[int] = None,
) -> Dict:
    """对多组 (n,m) 矩阵画：
    1) n 个子图（每个子图画 K 条曲线，对比 matrices[k][i,:]）
    2) 均值曲线（每组 mean），并可添加 std/stderr 阴影区间（fill_between）

    Parameters
    ----------
    matrices:
        - list/tuple: [mat1, mat2, ...]，每个 mat shape=(n,m)
        - dict: {"label": mat, ...}，label 直接用作图例
    labels:
        当 matrices 是 list/tuple 时可传；长度必须等于组数 K。
    ci:
        None 不画阴影；"std" 画 mean±std；"stderr" 画 mean±stderr。
    nan_policy:
        "propagate": 若存在 NaN 则统计结果可能变 NaN
        "omit": 使用 nanmean/nanstd，并按每列有效样本数计算 stderr
    max_cols:
        限制子图网格的最大列数（大 n 时可控图尺寸）。
    """

    # --------- normalize inputs ---------
    if isinstance(matrices, dict):
        _labels = list(matrices.keys())
        mats = [_as_2d_float(matrices[k], f"matrices['{k}']") for k in _labels]
    else:
        mats = [_as_2d_float(m, f"matrices[{i}]") for i, m in enumerate(matrices)]
        _labels = list(labels) if labels is not None else [f"M{i}" for i in range(len(mats))]

    if len(mats) < 2:
        raise ValueError(f"至少需要两组数据进行对比，但 got K={len(mats)}")
    if len(_labels) != len(mats):
        raise ValueError(f"labels 长度必须等于矩阵组数 K={len(mats)}，但 got len(labels)={len(_labels)}")

    n, m = mats[0].shape
    if n <= 0 or m <= 0:
        raise ValueError(f"矩阵 shape 必须是正数，但 got (n,m)=({n},{m})")
    for j, mat in enumerate(mats[1:], start=1):
        if mat.shape != (n, m):
            raise ValueError(
                f"所有矩阵形状必须一致，期望 (n,m)=({n},{m})，但 matrices[{j}].shape={mat.shape}"
            )

    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(m)

    # ---------------------------
    # 图1：n个子图（自动网格排版）
    # ---------------------------
    cols = math.ceil(math.sqrt(n))
    if max_cols is not None:
        cols = max(1, min(cols, int(max_cols)))
    rows = math.ceil(n / cols)

    fig1, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        ax = axes[i]
        for lab, mat in zip(_labels, mats):
            # 避免每个子图都重复 legend（可选）
            ax.plot(x, mat[i, :], label=(lab if show_subplot_legend else None))
        ax.set_title(f"Row i={i}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        if show_subplot_legend:
            ax.legend()

    for k in range(n, rows * cols):
        axes[k].axis("off")

    if per_row_title is None:
        per_row_title = "Per-row comparison: " + " vs ".join(_labels)
    fig1.suptitle(per_row_title, y=0.995)

    # 全局图例（默认更干净）
    if not show_subplot_legend:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        # axes[0] 上 label=None，所以需要从当前 rc cycle 的 line 拿 handle：改为从 fig 中找
        # 更稳：从最后一个有效子图里取所有 line
        src_ax = axes[0]
        for ax in axes[:n]:
            if len(ax.lines) >= len(mats):
                src_ax = ax
                break
        handles = src_ax.lines[: len(mats)]
        fig1.legend(handles, _labels, loc=global_legend_loc)

    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    fig1_png = os.path.join(save_dir, f"{prefix}_per_row.png")
    fig1.savefig(fig1_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig1_pdf = os.path.join(save_dir, f"{prefix}_per_row.pdf")
        fig1.savefig(fig1_pdf, bbox_inches="tight")

    # ---------------------------
    # 图2：均值曲线 + std/stderr 阴影
    # ---------------------------
    use_nan = nan_policy == "omit"
    mean_fn = np.nanmean if use_nan else np.mean
    std_fn = np.nanstd if use_nan else np.std

    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}
    stderrs: Dict[str, np.ndarray] = {}

    fig2 = plt.figure(figsize=(9.0, 4.4))
    ax2 = fig2.add_subplot(1, 1, 1)

    for lab, mat in zip(_labels, mats):
        mu = mean_fn(mat, axis=0)
        # n=1 时 ddof=1 会警告/产生 nan；这里自动退化 ddof=0
        ddof_eff = ddof if (n - ddof) > 0 else 0
        sd = std_fn(mat, axis=0, ddof=ddof_eff)

        if use_nan:
            cnt = np.sum(~np.isnan(mat), axis=0)
            se = np.where(cnt > 0, sd / np.sqrt(cnt), np.nan)
        else:
            se = sd / math.sqrt(n)

        means[lab] = mu
        stds[lab] = sd
        stderrs[lab] = se

        (line,) = ax2.plot(x, mu, label=lab)
        if ci is not None:
            ci_lower = ci.lower()
            if ci_lower not in {"std", "stderr"}:
                raise ValueError(f"ci 只能是 None/'std'/'stderr'，但 got ci={ci}")
            band = sd if ci_lower == "std" else se
            ax2.fill_between(
                x,
                mu - band,
                mu + band,
                alpha=ci_alpha,
                color=line.get_color(),
                linewidth=0,
            )

    if mean_title is None:
        if ci is None:
            mean_title = "Mean comparison: " + " vs ".join(_labels)
        else:
            mean_title = f"Mean ± {ci.lower()} comparison: " + " vs ".join(_labels)

    ax2.set_title(mean_title)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(f"mean {y_label}")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()

    fig2_png = os.path.join(save_dir, f"{prefix}_mean.png")
    fig2.savefig(fig2_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig2_pdf = os.path.join(save_dir, f"{prefix}_mean.pdf")
        fig2.savefig(fig2_pdf, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)

    return {
        "means": means,
        "stds": stds,
        "stderrs": stderrs,
        "saved": {
            "per_row_png": fig1_png,
            "mean_png": fig2_png,
            **({"per_row_pdf": fig1_pdf, "mean_pdf": fig2_pdf} if save_pdf else {}),
        },
    }


def plot_a_b_matrices(
    a: ArrayLike2D,
    b: ArrayLike2D,
    save_dir: str = ".",
    prefix: str = "gnn",
    dpi: int = 200,
    save_pdf: bool = False,
    label_1: str = "GNN",
    label_2: str = "NO_GNN",
    *,
    ci: Optional[str] = "stderr",
    ci_alpha: float = 0.18,
    ddof: int = 1,
    nan_policy: str = "propagate",
) -> Dict:
    """兼容旧接口：两组数据对比。"""

    out = plot_multi_matrices(
        [a, b],
        labels=[label_1, label_2],
        save_dir=save_dir,
        prefix=prefix,
        dpi=dpi,
        save_pdf=save_pdf,
        x_label="iteration",
        y_label="value",
        per_row_title=f"Per-row comparison: {label_1} vs {label_2}",
        mean_title=None,
        ci=ci,
        ci_alpha=ci_alpha,
        ddof=ddof,
        nan_policy=nan_policy,
    )

    # 兼容你之前的用法：保留 a_ave / b_ave 字段
    return {
        "a_ave": out["means"][label_1],
        "b_ave": out["means"][label_2],
        "a_std": out["stds"][label_1],
        "b_std": out["stds"][label_2],
        "a_stderr": out["stderrs"][label_1],
        "b_stderr": out["stderrs"][label_2],
        "saved": out["saved"],
        "_full": out,
    }


if __name__ == "__main__":
    # 示例：K=4 组，每组 shape=(n,m)
    n, m = 6, 30
    rng = np.random.default_rng(7)

    mats = [rng.standard_normal((n, m)).cumsum(axis=1) for _ in range(4)]
    labs = ["GNN", "NO_GNN", "SA", "GA"]

    out = plot_multi_matrices(
        mats,
        labels=labs,
        save_dir=".",
        prefix="demo_multi",
        ci="stderr",  # or "std" / None
        max_cols=4,
    )
    print("saved:", out["saved"])
