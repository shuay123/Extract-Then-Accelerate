import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


ArrayLike2D = Union[np.ndarray, Sequence[Sequence[float]]]


# =========================================================
# Helpers
# =========================================================

def _as_2d_float(x: ArrayLike2D, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，但 got {name}.ndim={arr.ndim}")
    return arr


def _ceil_sqrt_grid(k: int, max_cols: Optional[int] = None) -> Tuple[int, int]:
    if k <= 0:
        return 1, 1
    cols = int(math.ceil(math.sqrt(k)))
    if max_cols is not None:
        cols = max(1, min(cols, int(max_cols)))
    rows = int(math.ceil(k / cols))
    return rows, cols


def _safe_std_1d(vals: np.ndarray, ddof: int) -> float:
    """vals: 1d array without NaN"""
    if vals.size == 0:
        return float("nan")
    ddof_eff = ddof if vals.size > ddof else 0
    return float(np.std(vals, ddof=ddof_eff))


# =========================================================
# (A) 多组等长矩阵对比：每组 shape=(n,m)
# =========================================================

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
    nan_policy: str = "omit",  # "omit" | "propagate"
    show_subplot_legend: bool = False,
    global_legend_loc: str = "upper right",
    max_cols: Optional[int] = None,
) -> Dict:
    """对 K 组等长矩阵 (n,m) 画：
    1) n 个子图：第 i 个子图对比 matrices[k][i,:]
    2) 均值曲线：每组 mean(axis=0)，并可添加 std/stderr 阴影（fill_between）

    Parameters
    ----------
    matrices:
        - list/tuple: [mat1, mat2, ...]，每个 mat shape=(n,m)
        - dict: {"label": mat, ...}，label 直接用作图例
    ci:
        None 不画阴影；"std" 画 mean±std；"stderr" 画 mean±stderr。
    nan_policy:
        "omit" 使用 nanmean/nanstd；"propagate" 遇 NaN 会向下传播。
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
        raise ValueError(f"labels 长度必须等于组数 K={len(mats)}，但 got len(labels)={len(_labels)}")

    n, m = mats[0].shape
    if n <= 0 or m <= 0:
        raise ValueError(f"矩阵 shape 必须是正数，但 got (n,m)=({n},{m})")
    for j, mat in enumerate(mats[1:], start=1):
        if mat.shape != (n, m):
            raise ValueError(f"所有矩阵形状必须一致：期望 ({n},{m}) 但 matrices[{j}].shape={mat.shape}")

    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(m)

    # ---------------------------
    # 图1：n个子图（自动网格排版）
    # ---------------------------
    rows, cols = _ceil_sqrt_grid(n, max_cols=max_cols)

    fig1, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        ax = axes[i]
        for lab, mat in zip(_labels, mats):
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

    if not show_subplot_legend:
        # 找一个包含最多 lines 的子图来做全局图例
        src_ax = axes[0]
        best = -1
        for ax in axes[:n]:
            if len(ax.lines) > best:
                best = len(ax.lines)
                src_ax = ax
        handles = src_ax.lines[: len(mats)]
        fig1.legend(handles, _labels, loc=global_legend_loc)

    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    fig1_png = os.path.join(save_dir, f"{prefix}_per_row.png")
    fig1.savefig(fig1_png, dpi=dpi, bbox_inches="tight")
    fig1_pdf = None
    if save_pdf:
        fig1_pdf = os.path.join(save_dir, f"{prefix}_per_row.pdf")
        fig1.savefig(fig1_pdf, bbox_inches="tight")

    # ---------------------------
    # 图2：均值曲线 + std/stderr 阴影
    # ---------------------------
    use_nan = nan_policy.lower() == "omit"
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
    fig2_pdf = None
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
    nan_policy: str = "omit",
) -> Dict:
    """兼容你的旧接口：输入两组 (n,m) 矩阵。

    额外支持：均值曲线阴影 (ci=std/stderr)。
    """
    out = plot_multi_matrices(
        [a, b],
        labels=[label_1, label_2],
        save_dir=save_dir,
        prefix=prefix,
        dpi=dpi,
        save_pdf=save_pdf,
        ci=ci,
        ci_alpha=ci_alpha,
        ddof=ddof,
        nan_policy=nan_policy,
        per_row_title=f"Per-row comparison: {label_1} vs {label_2}",
        mean_title=None,
    )

    a_ave = out["means"][label_1]
    b_ave = out["means"][label_2]

    return {
        "a_ave": a_ave,
        "b_ave": b_ave,
        "a_std": out["stds"][label_1],
        "b_std": out["stds"][label_2],
        "a_stderr": out["stderrs"][label_1],
        "b_stderr": out["stderrs"][label_2],
        "saved": out["saved"],
    }


# =========================================================
# (B) 多组“时间序列点集”：每条序列是 (L,2)=[y,time]
#     - 每组可包含多条序列（重复 run/seed）
#     - 每条序列长度 L 可不同
# =========================================================

Series2D = Union[np.ndarray, Sequence[Sequence[float]]]
GroupSeries = Union[Series2D, Sequence[Series2D]]  # 单条序列 or 序列列表


def _coerce_group_to_runs(g: GroupSeries, name: str) -> List[np.ndarray]:
    """把 group 规范成: List[np.ndarray], 每个 run shape=(L,2+)"""
    # 单条序列：直接是 (L,2)
    if isinstance(g, np.ndarray):
        if g.ndim == 2:
            return [np.asarray(g, dtype=float)]
        if g.ndim == 3 and g.shape[-1] >= 2:
            # shape=(R,L,2)
            return [np.asarray(g[i], dtype=float) for i in range(g.shape[0])]
        raise ValueError(f"{name} 必须是 (L,2) 或 [(L,2), ...]，但 got ndarray.ndim={g.ndim}")

    # list/tuple：可能是单条序列(由点组成)，也可能是多条序列(list of arrays)
    if not isinstance(g, (list, tuple)):
        arr = np.asarray(g, dtype=float)
        if arr.ndim == 2:
            return [arr]
        raise ValueError(f"{name} 必须是二维序列或其列表")

    if len(g) == 0:
        return []

    # 判断 g 是“点列表”还是“run 列表”
    first = g[0]
    try:
        first_arr = np.asarray(first, dtype=float)
    except Exception:
        first_arr = None

    # 如果首元素本身就是二维数组 => 认为是 run 列表
    if isinstance(first_arr, np.ndarray) and first_arr.ndim == 2:
        runs = []
        for i, item in enumerate(g):
            arr = np.asarray(item, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(f"{name}[{i}] 必须是 shape=(L,2+) 的二维数组")
            runs.append(arr)
        return runs

    # 否则认为 g 本身就是单条序列（由点组成），整体转成 (L,2)
    arr = np.asarray(g, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} 必须能转换为 shape=(L,2+) 的二维数组")
    return [arr]


def _extract_xy_from_run(run: np.ndarray, *, y_col: int = 0, x_col: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """run: shape=(L,2+) 且列顺序默认 [y, x(time)]"""
    if run.ndim != 2 or run.shape[1] < 2:
        raise ValueError(f"每条 run 必须是 shape=(L,2+) 的二维数组，但 got {run.shape}")

    y = np.asarray(run[:, y_col], dtype=float)
    x = np.asarray(run[:, x_col], dtype=float)

    # 去掉 NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return x, y

    # 按 x 排序
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 合并重复 x：对 y 取平均
    if x.size > 1 and np.any(np.diff(x) == 0):
        ux, inv = np.unique(x, return_inverse=True)
        y_sum = np.zeros_like(ux, dtype=float)
        y_cnt = np.zeros_like(ux, dtype=float)
        for i, k in enumerate(inv):
            y_sum[k] += y[i]
            y_cnt[k] += 1
        y = y_sum / np.where(y_cnt > 0, y_cnt, 1)
        x = ux

    return x, y


def plot_multi_xy_groups(
    groups: Union[Sequence[GroupSeries], Dict[str, GroupSeries]],
    labels: Optional[Sequence[str]] = None,
    *,
    save_dir: str = ".",
    prefix: str = "ts",
    dpi: int = 200,
    save_pdf: bool = False,
    x_label: str = "time",
    y_label: str = "value",
    per_run_title: Optional[str] = None,
    mean_title: Optional[str] = None,
    ci: Optional[str] = "stderr",  # None | "std" | "stderr"
    ci_alpha: float = 0.18,
    ddof: int = 1,
    nan_policy: str = "omit",  # 这里只建议用 omit
    y_col: int = 0,
    x_col: int = 1,
    grid: Union[str, np.ndarray] = "auto",  # "auto" | "union" | "linspace" | np.ndarray
    grid_points: int = 200,
    max_unique_grid: int = 500,
    max_runs_plot: int = 25,
    show_subplot_legend: bool = False,
    global_legend_loc: str = "upper right",
    max_cols: Optional[int] = None,
) -> Dict:
    """多组“时间序列点集”对比。

    数据格式（你描述的情况）：
    - 每条序列(run) 是二维数组 shape=(L,2)，每行是一个点： [y, x_time]
    - 每组 group 可包含多条 run（重复实验/不同 seed），长度 L 可不同
    - 各组 run 数量也可不同

    本函数输出两张图：
    1) per-run 对比图：第 i 个子图对比每组的第 i 条 run（若该组不存在第 i 条则跳过）
    2) mean 曲线对比：先把每条 run 插值到统一的 x-grid，再计算 mean/std/stderr 并画阴影

    grid:
      - "auto": 若全局 unique x 数量不大且接近整数 -> 用 union；否则用 linspace(grid_points)
      - "union": 用全局所有 unique x 作为网格（超过 max_unique_grid 则退化为 linspace）
      - "linspace": 用 [xmin, xmax] 的等距网格
      - np.ndarray: 直接指定网格（1d）

    返回：
      out["grid"]: 统一 x-grid
      out["means"][label], out["stds"][label], out["stderrs"][label]
      out["saved"]: 保存路径
    """

    # --------- normalize inputs ---------
    if isinstance(groups, dict):
        _labels = list(groups.keys())
        group_runs = [_coerce_group_to_runs(groups[k], f"groups['{k}']") for k in _labels]
    else:
        group_list = list(groups)
        _labels = list(labels) if labels is not None else [f"G{i}" for i in range(len(group_list))]
        if len(_labels) != len(group_list):
            raise ValueError(f"labels 长度必须等于组数 K={len(group_list)}，但 got len(labels)={len(_labels)}")
        group_runs = [_coerce_group_to_runs(g, f"groups[{i}]") for i, g in enumerate(group_list)]

    if len(_labels) < 2:
        raise ValueError(f"至少需要两组数据进行对比，但 got K={len(_labels)}")

    os.makedirs(save_dir, exist_ok=True)

    # 抽取所有 run 的 (x,y)
    runs_xy: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    for lab, runs in zip(_labels, group_runs):
        tmp = []
        for r in runs:
            x, y = _extract_xy_from_run(np.asarray(r, dtype=float), y_col=y_col, x_col=x_col)
            tmp.append((x, y))
        runs_xy.append(tmp)

    # ---------------------------
    # 图1：per-run 子图（按 run index 对齐，不足则跳过）
    # ---------------------------
    max_runs = max((len(r) for r in runs_xy), default=0)
    runs_to_plot = min(max_runs, int(max_runs_plot))
    if runs_to_plot <= 0:
        raise ValueError("所有组都没有有效 run（空列表或全 NaN）")

    rows, cols = _ceil_sqrt_grid(runs_to_plot, max_cols=max_cols)
    fig1, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.4 * rows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for i in range(runs_to_plot):
        ax = axes[i]
        for lab, group in zip(_labels, runs_xy):
            if i >= len(group):
                continue
            x, y = group[i]
            if x.size == 0:
                continue
            ax.plot(x, y, label=(lab if show_subplot_legend else None))
        ax.set_title(f"Run i={i}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        if show_subplot_legend:
            ax.legend()

    for k in range(runs_to_plot, rows * cols):
        axes[k].axis("off")

    if per_run_title is None:
        per_run_title = "Per-run comparison (ragged): " + " vs ".join(_labels)
    fig1.suptitle(per_run_title, y=0.995)

    if not show_subplot_legend:
        # 找一个拥有最多 lines 的子图来做全局图例
        src_ax = axes[0]
        best = -1
        for ax in axes[:runs_to_plot]:
            if len(ax.lines) > best:
                best = len(ax.lines)
                src_ax = ax
        handles = src_ax.lines
        # handles 可能少于组数（因为某些组缺 run/为空），但 label 顺序仍按出现顺序
        # 这里更稳：根据 _labels 生成 proxy handles
        if len(handles) > 0:
            # 如果 handles 少，仍使用已有 handles 的颜色；其余用空 proxy
            proxy = []
            for j, lab in enumerate(_labels):
                if j < len(handles):
                    proxy.append(handles[j])
                else:
                    proxy.append(plt.Line2D([], [], linestyle="-"))
            fig1.legend(proxy, _labels, loc=global_legend_loc)

    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    fig1_png = os.path.join(save_dir, f"{prefix}_per_run.png")
    fig1.savefig(fig1_png, dpi=dpi, bbox_inches="tight")
    fig1_pdf = None
    if save_pdf:
        fig1_pdf = os.path.join(save_dir, f"{prefix}_per_run.pdf")
        fig1.savefig(fig1_pdf, bbox_inches="tight")

    # ---------------------------
    # 图2：mean + std/stderr 阴影（先插值到统一网格）
    # ---------------------------
    # 统一 x-grid：使用全局 xmin/xmax
    all_x = np.concatenate([x for grp in runs_xy for (x, _y) in grp if x.size > 0], axis=0)
    if all_x.size == 0:
        raise ValueError("所有 run 都没有有效 x")
    x_min = float(np.nanmin(all_x))
    x_max = float(np.nanmax(all_x))

    def _make_grid() -> np.ndarray:
        if isinstance(grid, np.ndarray):
            g = np.asarray(grid, dtype=float).reshape(-1)
            return np.sort(g)

        mode = str(grid).lower()
        if mode not in {"auto", "union", "linspace"}:
            raise ValueError(f"grid 只能是 'auto'/'union'/'linspace'/np.ndarray，但 got grid={grid}")

        # union 候选
        uniq = np.unique(all_x[~np.isnan(all_x)])
        uniq = np.sort(uniq)

        if mode in {"union", "auto"}:
            if uniq.size <= max_unique_grid:
                if mode == "union":
                    return uniq
                # auto: 若 x 接近整数（迭代/代数）就用 union，否则 linspace
                if np.allclose(uniq, np.round(uniq)):
                    return uniq

        # linspace
        pts = max(2, int(grid_points))
        return np.linspace(x_min, x_max, pts)

    xg = _make_grid()

    use_nan = nan_policy.lower() == "omit"

    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}
    stderrs: Dict[str, np.ndarray] = {}
    counts: Dict[str, np.ndarray] = {}

    fig2 = plt.figure(figsize=(9.4, 4.6))
    ax2 = fig2.add_subplot(1, 1, 1)

    for lab, grp in zip(_labels, runs_xy):
        if len(grp) == 0:
            means[lab] = np.full_like(xg, np.nan, dtype=float)
            stds[lab] = np.full_like(xg, np.nan, dtype=float)
            stderrs[lab] = np.full_like(xg, np.nan, dtype=float)
            counts[lab] = np.zeros_like(xg, dtype=int)
            continue

        Y = np.full((len(grp), xg.size), np.nan, dtype=float)

        for r, (x, y) in enumerate(grp):
            if x.size == 0:
                continue
            if x.size == 1:
                # 只有一个点：只在最接近的 grid 点落值
                j = int(np.argmin(np.abs(xg - x[0])))
                Y[r, j] = y[0]
                continue

            # 只在该 run 的有效范围内插值，其余保持 NaN
            mask = (xg >= x[0]) & (xg <= x[-1])
            if not np.any(mask):
                continue

            Y[r, mask] = np.interp(xg[mask], x, y)

        if use_nan:
            mu = np.nanmean(Y, axis=0)
            cnt = np.sum(~np.isnan(Y), axis=0)

            sd = np.full_like(mu, np.nan, dtype=float)
            for j in range(xg.size):
                vals = Y[:, j]
                vals = vals[~np.isnan(vals)]
                sd[j] = _safe_std_1d(vals, ddof)

            se = np.where(cnt > 0, sd / np.sqrt(cnt), np.nan)
        else:
            mu = np.mean(Y, axis=0)
            cnt = np.full_like(mu, fill_value=Y.shape[0], dtype=int)
            ddof_eff = ddof if (Y.shape[0] - ddof) > 0 else 0
            sd = np.std(Y, axis=0, ddof=ddof_eff)
            se = sd / math.sqrt(Y.shape[0])

        means[lab] = mu
        stds[lab] = sd
        stderrs[lab] = se
        counts[lab] = cnt

        (line,) = ax2.plot(xg, mu, label=lab)
        if ci is not None:
            ci_lower = ci.lower()
            if ci_lower not in {"std", "stderr"}:
                raise ValueError(f"ci 只能是 None/'std'/'stderr'，但 got ci={ci}")
            band = sd if ci_lower == "std" else se
            ax2.fill_between(
                xg,
                mu - band,
                mu + band,
                alpha=ci_alpha,
                color=line.get_color(),
                linewidth=0,
            )

    if mean_title is None:
        if ci is None:
            mean_title = "Mean comparison (aligned by time): " + " vs ".join(_labels)
        else:
            mean_title = f"Mean ± {ci.lower()} (aligned by time): " + " vs ".join(_labels)

    ax2.set_title(mean_title)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(f"mean {y_label}")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()

    fig2_png = os.path.join(save_dir, f"{prefix}_mean.png")
    fig2.savefig(fig2_png, dpi=dpi, bbox_inches="tight")
    fig2_pdf = None
    if save_pdf:
        fig2_pdf = os.path.join(save_dir, f"{prefix}_mean.pdf")
        fig2.savefig(fig2_pdf, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)

    return {
        "grid": xg,
        "means": means,
        "stds": stds,
        "stderrs": stderrs,
        "counts": counts,
        "saved": {
            "per_run_png": fig1_png,
            "mean_png": fig2_png,
            **({"per_run_pdf": fig1_pdf, "mean_pdf": fig2_pdf} if save_pdf else {}),
        },
    }


# =========================================================
# Quick demo
# =========================================================

if __name__ == "__main__":
    # demo 1: 等长矩阵
    n, m = 5, 20
    a = np.random.randn(n, m).cumsum(axis=1)
    b = np.random.randn(n, m).cumsum(axis=1)
    out = plot_a_b_matrices(a, b, save_dir=".", prefix="demo_ab", ci="stderr")
    print("demo_ab saved:", out["saved"])

    # demo 2: ragged time-series groups
    # 每条 run: (L,2) -> [y, t]
    g1 = [
        np.column_stack([np.random.randn(40).cumsum(), np.arange(40)]),
        np.column_stack([np.random.randn(30).cumsum(), np.arange(30)]),
        np.column_stack([np.random.randn(50).cumsum(), np.arange(50)]),
    ]
    g2 = [
        np.column_stack([np.random.randn(45).cumsum(), np.arange(45)]),
        np.column_stack([np.random.randn(35).cumsum(), np.arange(35)]),
    ]

    out2 = plot_multi_xy_groups(
        {"G1": g1, "G2": g2},
        save_dir=".",
        prefix="demo_ts",
        ci="stderr",
        grid="auto",
        grid_points=60,
        max_runs_plot=9,
    )
    print("demo_ts saved:", out2["saved"])
