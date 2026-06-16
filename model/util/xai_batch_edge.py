import os
import json
import random
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import shap

# =========================
# Flexible imports
# =========================
try:
    from model.model import UndirectedContrastiveClusteringModel
except Exception:
    from model import UndirectedContrastiveClusteringModel

try:
    from utils.DatasetGennerate import get_dataset_big_shedule, get_dataset
except Exception:
    try:
        from DatasetGennerate import get_dataset_big_shedule, get_dataset
    except Exception:
        get_dataset_big_shedule = None
        from DatasetGennerate import get_dataset

EPS = 1e-8
SEED = 42

warnings.filterwarnings(
    "ignore",
    message="The NumPy global RNG was seeded by calling `np.random.seed`"
)


# =========================
# Basic setup
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# Config loading
# =========================
def load_yaml_config_flexible(yaml_path):
    try:
        from util.config_loader import load_yaml_config
        return load_yaml_config(yaml_path)
    except Exception:
        pass

    try:
        from utils.config_loader_gnn import load_yaml_config
        return load_yaml_config(yaml_path)
    except Exception:
        pass

    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if not isinstance(obj, dict):
            raise ValueError("YAML content is not a dict.")
        return obj
    except Exception as e:
        raise RuntimeError(
            f"Unable to load yaml config from {yaml_path}. "
            f"Please ensure util.config_loader or utils.config_loader_gnn is available. Original error: {e}"
        )


# =========================
# Model loading
# =========================
def resolve_schedule_ckpt(args, ckpt_path=None):
    if ckpt_path:
        return ckpt_path

    n_batch = args.get("n_batch", 0)
    n_nodes = args.get("n_nodes", 0)
    candidates = [
        f"seruschedule_W{n_batch}_J{n_nodes}_pre.pt",
        f"seruschedule_W{n_batch}_J{n_nodes}_f1.pt",
        os.path.join("models_trained", f"seruschedule_W{n_batch}_J{n_nodes}_pre.pt"),
        os.path.join("models_trained", f"seruschedule_W{n_batch}_J{n_nodes}_f1.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def build_model_from_yaml_and_ckpt(args, ckpt_path):
    # IMPORTANT: mirror main_schedule.py, where n = n_batch * 4
    args_n = args.get("n_batch", 0) * 4
    model = UndirectedContrastiveClusteringModel(
        m=args["m"],
        n=args_n,
        n_nodes=args["n_nodes"],
        hidden_dim=args["hidden_dim"],
        n_gcn_layers=args["n_gcn_layers"],
        temperature=args["temperature"],
        use_contrastive=args["use_contrastive"],
    )

    state_obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_obj, dict) and "state_dict" in state_obj:
        state_dict = state_obj["state_dict"]
    else:
        state_dict = state_obj
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# =========================
# Optional explain-mode forward
# =========================
def model_forward_for_xai(model, node_features):
    """
    Prefer return_explain=True.
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
    edge_gen = getattr(model, "edge_feature_generator", None)
    if edge_gen is None:
        return None
    if hasattr(edge_gen, "product_weights") and hasattr(edge_gen, "process_weights"):
        pw = edge_gen.product_weights.detach().cpu().numpy()
        qw = edge_gen.process_weights.detach().cpu().numpy()
        return np.outer(pw, qw).reshape(-1).astype(np.float32)
    return None


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


def build_similarity_matrix(batch_mat, w):
    n_nodes = batch_mat.shape[0]
    sim_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                sim_mat[i, j] = np.nan
            else:
                sim_mat[i, j] = weighted_similarity(batch_mat[i], batch_mat[j], w)
    return sim_mat


def percentile_rank(value, values):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values <= value))


# =========================
# Directed tensor rebuilding
# =========================
def rebuild_directed_tensor(edge_arr, n_nodes):
    """
    edge_arr: [N*(N-1), C] or [N*(N-1)]
    return: [N, N, C] or [N, N]
    """
    if edge_arr.ndim == 1:
        full = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        mask = ~np.eye(n_nodes, dtype=bool)
        full.reshape(-1)[mask.reshape(-1)] = edge_arr
        return full

    c = edge_arr.shape[-1]
    full = np.zeros((n_nodes, n_nodes, c), dtype=np.float32)
    mask = ~np.eye(n_nodes, dtype=bool)
    full.reshape(-1, c)[mask.reshape(-1)] = edge_arr
    return full


def build_similarity_matrix_from_raw(raw_edge_full):
    n_nodes = raw_edge_full.shape[0]
    sim_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                sim_mat[i, j] = np.nan
            else:
                sim_mat[i, j] = float(0.5 * (raw_edge_full[i, j, 0] + raw_edge_full[j, i, 0]))
    return sim_mat


def build_batch_pair_features(batch_mat, i, j, sim_mat, raw_edge_full=None, w=None):
    xi = batch_mat[i]
    xj = batch_mat[j]

    if raw_edge_full is not None:
        raw_sym = 0.5 * (raw_edge_full[i, j] + raw_edge_full[j, i])
        f1_sim = float(raw_sym[0])
        f2_comp = float(raw_sym[1])
        f3_overlap = float(raw_sym[2])
        f4_gap = float(raw_sym[3])
    else:
        if w is None:
            w = np.ones_like(xi, dtype=np.float32)
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
def build_latent_summary_features(edge_emb_full, i, j):
    e_sym = 0.5 * (edge_emb_full[i, j] + edge_emb_full[j, i])
    return {
        "f11_emb_norm": float(np.linalg.norm(e_sym)),
        "f12_emb_mean": float(np.mean(e_sym)),
        "f13_emb_std": float(np.std(e_sym)),
    }


# =========================
# Data export
# =========================
def extract_batch_edge_dataframe_v1(model, dataloader, device, split_name="data", max_batches=None):
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
            has_raw_edge = (
                explain_dict is not None
                and isinstance(explain_dict, dict)
                and "raw_edge_features" in explain_dict
                and explain_dict["raw_edge_features"] is not None
            )

            edge_emb_np = explain_dict["edge_emb"].detach().cpu().numpy() if has_edge_emb else None
            raw_edge_np = explain_dict["raw_edge_features"].detach().cpu().numpy() if has_raw_edge else None

            bsz, n_nodes, feat_dim = node_np.shape

            for b in range(bsz):
                batch_mat = node_np[b]

                if has_raw_edge:
                    raw_edge_full = rebuild_directed_tensor(raw_edge_np[b], n_nodes)
                    sim_mat = build_similarity_matrix_from_raw(raw_edge_full)
                else:
                    raw_edge_full = None
                    if w is None:
                        w_eff = np.ones(feat_dim, dtype=np.float32)
                    else:
                        w_eff = w
                    sim_mat = build_similarity_matrix(batch_mat, w_eff)

                if has_edge_emb:
                    edge_emb_full = rebuild_directed_tensor(edge_emb_np[b], n_nodes)
                else:
                    edge_emb_full = None

                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        score_sym = float(scores_np[b, i, j])

                        feat_dict = build_batch_pair_features(
                            batch_mat=batch_mat,
                            i=i,
                            j=j,
                            sim_mat=sim_mat,
                            raw_edge_full=raw_edge_full,
                            w=w,
                        )

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
                            "batch_i": i,
                            "batch_j": j,
                            "label": int(labels_np[b, i, j]),
                            "score_sym": score_sym,
                            **feat_dict,
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


def fit_surrogate_v1(df_train, df_val, df_test, feature_cols):
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
        random_state=SEED,
    )

    surrogate.fit(df_train[feature_cols], df_train[target_col])

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
def save_bar_plot(shap_values, x_df, out_path):
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, x_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_beeswarm_plot(shap_values, x_df, out_path):
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, x_df, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plot(shap_values, x_df, feat_name, out_path):
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(feat_name, shap_values, x_df, show=False)
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


def save_waterfall_plot(explainer, shap_values, x_df, row_idx, out_path, max_display=10):
    if row_idx is None:
        return
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[row_idx],
            base_values=explainer.expected_value,
            data=x_df.iloc[row_idx],
            feature_names=x_df.columns.tolist(),
        ),
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Dataloader dispatch
# =========================
def build_schedule_dataloaders(args):
    if get_dataset is not None:
        return get_dataset(args, SizeofDataset=args.get("SizeofDataset", 5000))
    return get_dataset(args, SizeofDataset=args.get("SizeofDataset", 5000))


# =========================
# Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Batch-edge XAI for seru scheduling GNN")
    parser.add_argument("--yaml_path", type=str, default="JCompany_W8_J8_5000_schedule.yaml")
    parser.add_argument("--ckpt_path", type=str, default="models_trained/seruschedule_W8_J8_pre.pt")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    set_seed()

    args = load_yaml_config_flexible(cli_args.yaml_path)
    ckpt_path = resolve_schedule_ckpt(args, cli_args.ckpt_path)

    yaml_stem = Path(cli_args.yaml_path).stem
    out_dir = cli_args.out_dir or os.path.join(".", "xai_batch_edge_out_v1", yaml_stem)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_n = args.get("n_batch", 0) * 4

    print("=" * 80)
    print("Batch-edge XAI v1 for Seru Scheduling Model")
    print("=" * 80)
    print(f"yaml_path: {cli_args.yaml_path}")
    print(f"ckpt_path: {ckpt_path}")
    print(f"n_nodes(batch nodes): {args['n_nodes']}, n_batch(worker count proxy): {args['n_batch']}, m: {args['m']}, n_for_model: {args_n}")
    print(f"hidden_dim: {args['hidden_dim']}, n_gcn_layers: {args['n_gcn_layers']}")
    print("=" * 80)

    train_loader, val_loader, test_loader = build_schedule_dataloaders(args)
    model = build_model_from_yaml_and_ckpt(args, ckpt_path).to(device)

    # export all graphs, then split for surrogate
    df_train = extract_batch_edge_dataframe_v1(model, train_loader, device, split_name="train", max_batches=cli_args.max_batches)
    df_val = extract_batch_edge_dataframe_v1(model, val_loader, device, split_name="val", max_batches=cli_args.max_batches)
    df_test = extract_batch_edge_dataframe_v1(model, test_loader, device, split_name="test", max_batches=cli_args.max_batches)
    df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

    csv_path = os.path.join(out_dir, "batch_edge_explain_v1.csv")
    df_all.to_csv(csv_path, index=False)

    feature_cols, interpretable_cols, latent_cols = get_feature_columns(df_all)
    df_train_surr, df_val_surr, df_test_surr = split_by_graph(df_all)

    surrogate, metrics = fit_surrogate_v1(df_train_surr, df_val_surr, df_test_surr, feature_cols)

    metadata = {
        "yaml_path": cli_args.yaml_path,
        "ckpt_path": ckpt_path,
        "out_dir": out_dir,
        "feature_cols": feature_cols,
        "interpretable_cols": interpretable_cols,
        "latent_cols": latent_cols,
        "raw_initial_edge_features_used_for_f1_f4": True,
        "raw_edge_feature_fallback_to_recomputed": False,
        "notes": [
            "Nodes are treated as batches in the scheduling-side graph.",
            "label=1 means the two batches belong to the same seru in the ground-truth solution.",
            "score_sym is the GNN predicted symmetric edge score.",
            "f1-f4 are taken from raw initial edge features returned by model.forward(return_explain=True) when available.",
        ],
    }

    metrics_path = os.path.join(out_dir, "batch_surrogate_metrics_v1.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "metadata": metadata}, f, indent=2, ensure_ascii=False)

    # SHAP on full feature set
    x_test = df_test_surr[feature_cols]
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(x_test)

    save_bar_plot(shap_values, x_test, os.path.join(out_dir, "shap_bar_batch_v1_full.png"))
    save_beeswarm_plot(shap_values, x_test, os.path.join(out_dir, "shap_beeswarm_batch_v1_full.png"))

    # SHAP on interpretable subset
    idx_map = [feature_cols.index(c) for c in interpretable_cols]
    shap_values_interp = shap_values[:, idx_map]
    x_test_interp = x_test[interpretable_cols]

    save_bar_plot(shap_values_interp, x_test_interp, os.path.join(out_dir, "shap_bar_batch_v1_interpretable.png"))
    save_beeswarm_plot(shap_values_interp, x_test_interp, os.path.join(out_dir, "shap_beeswarm_batch_v1_interpretable.png"))

    for feat in ["f2_comp", "f10_mutual_rank", "f9_pair_excl", "f4_gap", "f1_sim"]:
        if feat in interpretable_cols:
            save_dependence_plot(
                shap_values_interp,
                x_test_interp,
                feat,
                os.path.join(out_dir, f"shap_dependence_{feat}_batch_v1.png"),
            )

    local_cases = pick_local_cases(df_test_surr)
    for name, idx in local_cases.items():
        save_waterfall_plot(
            explainer,
            shap_values,
            x_test,
            idx,
            os.path.join(out_dir, f"waterfall_{name}_batch_v1_full.png"),
        )

    print("=" * 80)
    print("Batch-edge XAI v1 finished.")
    print(f"CSV saved to: {csv_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Using latent summaries: {len(latent_cols) > 0}")
    print("Fidelity metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
