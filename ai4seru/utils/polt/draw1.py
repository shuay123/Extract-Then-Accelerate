import math
import numpy as np
import matplotlib.pyplot as plt


def plot_a_b_matrices(a, b, save_dir=".", prefix="gnn", dpi=200, save_pdf=False, label_1 = "GNN", label_2 = "NO_GNN"):
    """
    a, b: shape (n, m) 的二维矩阵（list of lists / numpy array 均可）
    产出：
      1) n 个子图对比 a[i,:] vs b[i,:]
      2) 均值曲线对比 a_ave vs b_ave
    并保存图片到磁盘
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a 和 b 必须是二维矩阵，但 got a.ndim={a.ndim}, b.ndim={b.ndim}")
    if a.shape != b.shape:
        raise ValueError(f"a 和 b 形状必须一致，但 got a.shape={a.shape}, b.shape={b.shape}")

    n, m = a.shape
    x = np.arange(m)

    # ---------------------------
    # 图1：n个子图（自动网格排版）
    # ---------------------------
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig1, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)  # 统一成一维，方便索引

    for i in range(n):
        ax = axes[i]
        ax.plot(x, a[i, :], label= label_1)
        ax.plot(x, b[i, :], label= label_2)
        ax.set_title(f"Row i={i}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        ax.legend()

    # 多出来的空子图隐藏掉
    for k in range(n, rows * cols):
        axes[k].axis("off")

    fig1.suptitle("Per-row comparison: GNN vs NO_GNN", y=0.995)
    fig1.tight_layout()

    fig1_png = f"{save_dir}/{prefix}_per_row.png"
    fig1.savefig(fig1_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig1_pdf = f"{save_dir}/{prefix}_per_row.pdf"
        fig1.savefig(fig1_pdf, bbox_inches="tight")

    # ---------------------------
    # 图2：按 n 维求均值，画均值曲线
    # ---------------------------
    a_ave = a.mean(axis=0)  # shape (m,)
    b_ave = b.mean(axis=0)  # shape (m,)

    fig2 = plt.figure(figsize=(8.5, 4.2))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x, a_ave, label= label_1 + "_ave")
    ax2.plot(x, b_ave, label= label_2 + "_ave")
    ax2.set_title("Mean over Cmax: GNN_ave vs NO_GNN_ave")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("mean Cmax")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()

    fig2_png = f"{save_dir}/{prefix}_mean.png"
    fig2.savefig(fig2_png, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig2_pdf = f"{save_dir}/{prefix}_mean.pdf"
        fig2.savefig(fig2_pdf, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)

    return {
        "a_ave": a_ave,
        "b_ave": b_ave,
        "saved": {
            "per_row_png": fig1_png,
            "mean_png": fig2_png,
            **({"per_row_pdf": fig1_pdf, "mean_pdf": fig2_pdf} if save_pdf else {})
        }
    }


# ---------------------------
# Usage example
# ---------------------------
if __name__ == "__main__":
    # 示例数据：n=5, m=20
    n, m = 5, 20
    a = np.random.randn(n, m).cumsum(axis=1)
    b = np.random.randn(n, m).cumsum(axis=1)

    out = plot_a_b_matrices(a, b, save_dir=".", prefix="demo", dpi=200, save_pdf=False)
    print("a_ave:", out["a_ave"])
    print("b_ave:", out["b_ave"])
    print("saved:", out["saved"])
