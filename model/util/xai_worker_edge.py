import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import shap

from model.model import UndirectedContrastiveClusteringModel
from utils.DatasetGennerate import get_dataset, get_dataset_big_config
from utils.config_loader_gnn import load_yaml_config


# =========================
# Basic setup
# =========================
EPS = 1e-8
SEED = 42

warnings.filterwarnings(
    "ignore",
    message="The NumPy global RNG was seeded by calling `np.random.seed`"
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# Model loading
# =========================
def build_model_from_yaml_and_ckpt(args, ckpt_path):
    model = UndirectedContrastiveClusteringModel(
        m=args["m"],
        n=args["n"],
        n_nodes=args["n_nodes"],
        hidden_dim=args["hidden_dim"],
        n_gcn_layers=args["n_gcn_layers"],
        temperature=args["temperature"],
        use_contrastive=args["use_contrastive"]
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# =========================
# Optional explain-mode forward
# =========================
def model_forward_for_xai(model, node_features):
    """
    优先尝试 return_explain=True。
    如果 model.py 还没打 explain 补丁，则自动回退。
    Returns:
        edge_scores: [B, N, N]
        explain_dict or None
    """
    try:
        out = model(node_features, labels=None, epoch=0, return_explain=True)
        if isinstance(out, tuple) and len(out) == 3:
            edge_scores, _, explain_dict = out
            return edge_scores, explain_dict
    except TypeError:
        pass
    except Exception:
        pass

    out = model(node_features)
    if isinstance(out, tuple):
        edge_scores = out[0]
    else:
        edge_scores = out
    return edge_scores, None


# =========================
# Weight vector from trained edge generator
# =========================
def get_learned_weight_vector(model):
    edge_gen = model.edge_feature_generator
    pw = edge_gen.product_weights.detach().cpu().numpy()   # [m]
    qw = edge_gen.process_weights.detach().cpu().numpy()   # [n]
    w = np.outer(pw, qw).reshape(-1).astype(np.float32)
    return w


# =========================
# Interpretable feature engineering
# =========================
def weighted_similarity(xi, xj, w):
    return float(np.exp(-np.linalg.norm((xi - xj) * w)))


def normalized_shared_capability(xi, xj, w):
    num = np.sum(np.minimum(xi, xj) * w)
    den = 0.5 * (np.sum(xi * w) + np.sum(xj * w)) + EPS
    return float(num / den)


def top_quartile_mask(x):
    x = np.asarray(x)
    pos = x[x > 0]
    if len(pos) == 0:
        return np.zeros_like(x, dtype=bool)
    q = np.quantile(pos, 0.75)
    return (x >= q) & (x > 0)


def weighted_top_overlap(xi, xj, w):
    mi = top_quartile_mask(xi)
    mj = top_quartile_mask(xj)
    inter = np.sum(w[(mi & mj)])
    union = np.sum(w[(mi | mj)]) + EPS
    return float(inter / union)


def normalized_total_gap(xi, xj):
    si = np.sum(xi)
    sj = np.sum(xj)
    return float(np.abs(si - sj) / (si + sj + EPS))


def build_similarity_matrix(worker_mat, w):
    N = worker_mat.shape[0]
    sim_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                sim_mat[i, j] = np.nan
            else:
                sim_mat[i, j] = weighted_similarity(worker_mat[i], worker_mat[j], w)
    return sim_mat


def percentile_rank(value, values):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values <= value))


def build_pair_features_v3(worker_mat, i, j, w, sim_mat):
    xi = worker_mat[i]
    xj = worker_mat[j]

    # 10 interpretable features
    f1_sim = weighted_similarity(xi, xj, w)
    f2_comp = normalized_shared_capability(xi, xj, w)
    f3_overlap = weighted_top_overlap(xi, xj, w)
    f4_gap = normalized_total_gap(xi, xj)

    row_i = np.delete(sim_mat[i], i)
    row_j = np.delete(sim_mat[j], j)

    f5_ctx_i_mean = float(np.nanmean(row_i))
    f6_ctx_j_mean = float(np.nanmean(row_j))
    f7_ctx_i_std = float(np.nanstd(row_i))
    f8_ctx_j_std = float(np.nanstd(row_j))

    f9_pair_excl = float(f1_sim - 0.5 * (f5_ctx_i_mean + f6_ctx_j_mean))

    r_i_j = percentile_rank(f1_sim, row_i)
    r_j_i = percentile_rank(f1_sim, row_j)
    f10_mutual_rank = float(0.5 * (r_i_j + r_j_i))

    return {
        "f1_sim": f1_sim,
        "f2_comp": f2_comp,
        "f3_overlap": f3_overlap,
        "f4_gap": f4_gap,
        "f5_ctx_i_mean": f5_ctx_i_mean,
        "f6_ctx_j_mean": f6_ctx_j_mean,
        "f7_ctx_i_std": f7_ctx_i_std,
        "f8_ctx_j_std": f8_ctx_j_std,
        "f9_pair_excl": f9_pair_excl,
        "f10_mutual_rank": f10_mutual_rank,
    }


# =========================
# Optional latent summary features
# =========================
def rebuild_directed_embedding_tensor(edge_emb, n_nodes):
    """
    edge_emb: [N*(N-1), H]
    return: [N, N, H]
    """
    H = edge_emb.shape[-1]
    full = np.zeros((n_nodes, n_nodes, H), dtype=np.float32)
    mask = ~np.eye(n_nodes, dtype=bool)
    full.reshape(-1, H)[mask.reshape(-1)] = edge_emb
    return full


def build_latent_summary_features(edge_emb_full, i, j):
    """
    对无向边(i,j)，取 e_ij 与 e_ji 平均后做摘要
    """
    e_sym = 0.5 * (edge_emb_full[i, j] + edge_emb_full[j, i])
    return {
        "f11_emb_norm": float(np.linalg.norm(e_sym)),
        "f12_emb_mean": float(np.mean(e_sym)),
        "f13_emb_std": float(np.std(e_sym)),
    }


# =========================
# Data export
# =========================
def extract_worker_edge_dataframe_v3(model, dataloader, device, split_name="data", max_batches=None):
    rows = []
    model.eval()

    w = get_learned_weight_vector(model)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                node_features, labels = batch
            else:
                raise ValueError("Dataloader batch format must be (node_features, labels).")

            node_features = node_features.to(device)
            labels = labels.to(device)

            edge_scores, explain_dict = model_forward_for_xai(model, node_features)

            scores_np = edge_scores.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            node_np = node_features.cpu().numpy()

            has_edge_emb = (
                explain_dict is not None
                and isinstance(explain_dict, dict)
                and "edge_emb" in explain_dict
                and explain_dict["edge_emb"] is not None
            )

            edge_emb_np = explain_dict["edge_emb"].detach().cpu().numpy() if has_edge_emb else None

            B, N, D = node_np.shape

            for b in range(B):
                worker_mat = node_np[b]
                sim_mat = build_similarity_matrix(worker_mat, w)

                if has_edge_emb:
                    edge_emb_full = rebuild_directed_embedding_tensor(edge_emb_np[b], N)
                else:
                    edge_emb_full = None

                for i in range(N):
                    for j in range(i + 1, N):
                        score_sym = float(scores_np[b, i, j])

                        feat_dict = build_pair_features_v3(worker_mat, i, j, w, sim_mat)

                        if edge_emb_full is not None:
                            feat_dict.update(build_latent_summary_features(edge_emb_full, i, j))
                        else:
                            feat_dict.update({
                                "f11_emb_norm": np.nan,
                                "f12_emb_mean": np.nan,
                                "f13_emb_std": np.nan,
                            })

                        rows.append({
                            "graph_id": f"{split_name}_{batch_idx}_{b}",
                            "i": i,
                            "j": j,
                            "label": int(labels_np[b, i, j]),
                            "score_sym": score_sym,
                            **feat_dict
                        })

    return pd.DataFrame(rows)


# =========================
# Split and surrogate
# =========================
def split_by_graph(df):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    idx_train, idx_temp = next(gss1.split(df, groups=df["graph_id"]))

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_temp = df.iloc[idx_temp].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1)
    idx_val, idx_test = next(gss2.split(df_temp, groups=df_temp["graph_id"]))

    df_val = df_temp.iloc[idx_val].reset_index(drop=True)
    df_test = df_temp.iloc[idx_test].reset_index(drop=True)

    return df_train, df_val, df_test


def get_feature_columns(df):
    base_cols = [
        "f1_sim",
        "f2_comp",
        "f3_overlap",
        "f4_gap",
        "f5_ctx_i_mean",
        "f6_ctx_j_mean",
        "f7_ctx_i_std",
        "f8_ctx_j_std",
        "f9_pair_excl",
        "f10_mutual_rank",
    ]
    latent_cols = ["f11_emb_norm", "f12_emb_mean", "f13_emb_std"]

    use_latent = not df[latent_cols].isna().all().all()
    if use_latent:
        return base_cols + latent_cols, base_cols, latent_cols
    return base_cols, base_cols, []


def fit_surrogate_v3(df_train, df_val, df_test, feature_cols):
    # raw score target，不再用 pseudo-logit
    target_col = "score_sym"

    surrogate = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.025,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.1,
        tree_method="hist",
        random_state=SEED
    )

    surrogate.fit(
        df_train[feature_cols],
        df_train[target_col]
    )

    pred_val = surrogate.predict(df_val[feature_cols])
    pred_test = surrogate.predict(df_test[feature_cols])

    metrics = {
        "val_r2": float(r2_score(df_val[target_col], pred_val)),
        "val_mae": float(mean_absolute_error(df_val[target_col], pred_val)),
        "val_spearman": float(spearmanr(df_val[target_col], pred_val).correlation),
        "test_r2": float(r2_score(df_test[target_col], pred_test)),
        "test_mae": float(mean_absolute_error(df_test[target_col], pred_test)),
        "test_spearman": float(spearmanr(df_test[target_col], pred_test).correlation),
    }

    return surrogate, metrics


# =========================
# SHAP plotting
# =========================
def save_bar_plot(shap_values, X, out_path):
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_beeswarm_plot(shap_values, X, out_path):
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plot(shap_values, X, feat_name, out_path):
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(feat_name, shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def pick_local_cases(df_test):
    pos = df_test[df_test["label"] == 1].copy()
    neg = df_test[df_test["label"] == 0].copy()

    tp_idx = pos["score_sym"].idxmax() if len(pos) > 0 else None
    fn_idx = pos["score_sym"].idxmin() if len(pos) > 0 else None
    fp_idx = neg["score_sym"].idxmax() if len(neg) > 0 else None
    tn_idx = neg["score_sym"].idxmin() if len(neg) > 0 else None

    return {"tp": tp_idx, "tn": tn_idx, "fp": fp_idx, "fn": fn_idx}


def save_waterfall_plot(explainer, shap_values, X, row_idx, out_path, max_display=10):
    if row_idx is None:
        return
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[row_idx],
            base_values=explainer.expected_value,
            data=X.iloc[row_idx],
            feature_names=X.columns.tolist(),
        ),
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
def main():
    set_seed()

    ckpt_path = "models_trained/seruconfig_W25_J200_pre.pt"
    yaml_path = "JCompany_W25_J200_5000.yaml"
    out_dir = "./xai_worker_edge_out_v3/config/W25J200"

    ensure_dir(out_dir)

    args = load_yaml_config(yaml_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Worker-edge XAI v3 for Seru Formation Model")
    print("=" * 80)
    print(f"yaml_path: {yaml_path}")
    print(f"ckpt_path: {ckpt_path}")
    print(f"n_nodes: {args['n_nodes']}, n_batch: {args['n_batch']}, m: {args['m']}, n: {args['n']}")
    print(f"hidden_dim: {args['hidden_dim']}, n_gcn_layers: {args['n_gcn_layers']}")
    print("=" * 80)

    train_loader, val_loader, test_loader = get_dataset_big_config(
        args, SizeofDataset=args.get("SizeofDataset", 5000)
    )

    model = build_model_from_yaml_and_ckpt(args, ckpt_path).to(device)

    # 导出全部图，再做 surrogate 切分
    df_train = extract_worker_edge_dataframe_v3(model, train_loader, device, split_name="train")
    df_val = extract_worker_edge_dataframe_v3(model, val_loader, device, split_name="val")
    df_test = extract_worker_edge_dataframe_v3(model, test_loader, device, split_name="test")
    df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

    csv_path = os.path.join(out_dir, "worker_edge_explain_v3.csv")
    df_all.to_csv(csv_path, index=False)

    feature_cols, interpretable_cols, latent_cols = get_feature_columns(df_all)

    df_train_surr, df_val_surr, df_test_surr = split_by_graph(df_all)

    surrogate, metrics = fit_surrogate_v3(
        df_train_surr, df_val_surr, df_test_surr, feature_cols
    )

    metrics_path = os.path.join(out_dir, "worker_surrogate_metrics_v3.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # SHAP on full feature set
    X_test = df_test_surr[feature_cols]
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X_test)

    save_bar_plot(
        shap_values, X_test,
        os.path.join(out_dir, "shap_bar_worker_v3_full.png")
    )
    save_beeswarm_plot(
        shap_values, X_test,
        os.path.join(out_dir, "shap_beeswarm_worker_v3_full.png")
    )

    # SHAP on interpretable subset only (方便主文使用)
    idx_map = [feature_cols.index(c) for c in interpretable_cols]
    shap_values_interp = shap_values[:, idx_map]
    X_test_interp = X_test[interpretable_cols]

    save_bar_plot(
        shap_values_interp, X_test_interp,
        os.path.join(out_dir, "shap_bar_worker_v3_interpretable.png")
    )
    save_beeswarm_plot(
        shap_values_interp, X_test_interp,
        os.path.join(out_dir, "shap_beeswarm_worker_v3_interpretable.png")
    )

    for feat in ["f2_comp", "f10_mutual_rank", "f9_pair_excl", "f4_gap", "f1_sim"]:
        if feat in interpretable_cols:
            idx = interpretable_cols.index(feat)
            save_dependence_plot(
                shap_values_interp, X_test_interp, feat,
                os.path.join(out_dir, f"shap_dependence_{feat}_worker_v3.png")
            )

    local_cases = pick_local_cases(df_test_surr)
    for name, idx in local_cases.items():
        save_waterfall_plot(
            explainer, shap_values, X_test, idx,
            os.path.join(out_dir, f"waterfall_{name}_worker_v3_full.png")
        )

    print("=" * 80)
    print("Worker-edge XAI v3 finished.")
    print(f"CSV saved to: {csv_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Using latent summaries: {len(latent_cols) > 0}")
    print("Fidelity metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()