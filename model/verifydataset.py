import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.DatasetGennerate import ClusteringDataset, MeaningfulClusteringDataset, RealisticClusteringDataset, SemanticClusteringDataset

def verify_dataset_quality(dataset, device='cpu'):
    """验证数据集是否可学习"""
    
    print("="*60)
    print("数据集质量检查")
    print("="*60)
    
    # 随机采样一个batch
    sample_features, sample_labels = dataset[0]
    
    # 1. 检查特征的聚类结构
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    X = sample_features.numpy()
    y_true = []
    n_nodes = sample_labels.shape[0]
    
    # 从标签矩阵提取真实的cluster assignment
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if sample_labels[i, j] == 1:
                # 找到节点i所属的cluster
                cluster_id = None
                for c_id in range(n_nodes):
                    if sample_labels[i, c_id] == 1:
                        cluster_id = c_id
                        break
                y_true.append(cluster_id)
    
    # 简化：直接用KMeans聚类，看是否能恢复真实标签
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # 计算轮廓系数（衡量聚类质量）
    silhouette = silhouette_score(X, y_pred)
    
    print(f"\n1. 特征聚类质量:")
    print(f"   轮廓系数: {silhouette:.4f}")
    print(f"   解释: >0.5优秀, 0.25-0.5合理, <0.25差")
    
    if silhouette < 0.25:
        print(f"   ❌ 警告：特征没有明显的聚类结构！")
    else:
        print(f"   ✓ 特征具有可学习的聚类结构")
    
    # 2. 检查同cluster节点的特征相似度
    same_cluster_dists = []
    diff_cluster_dists = []
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = torch.norm(sample_features[i] - sample_features[j]).item()
            
            if sample_labels[i, j] == 1:
                same_cluster_dists.append(dist)
            else:
                diff_cluster_dists.append(dist)
    
    avg_same = np.mean(same_cluster_dists)
    avg_diff = np.mean(diff_cluster_dists)
    separation_ratio = avg_diff / (avg_same + 1e-8)
    
    print(f"\n2. 簇内/簇间距离:")
    print(f"   簇内平均距离: {avg_same:.4f}")
    print(f"   簇间平均距离: {avg_diff:.4f}")
    print(f"   分离比率: {separation_ratio:.4f}")
    print(f"   解释: >1.5优秀, 1.2-1.5合理, <1.2差")
    
    if separation_ratio < 1.2:
        print(f"   ❌ 警告：簇间距离太小，难以区分！")
    else:
        print(f"   ✓ 簇间有明显分离")
    
    # 3. 检查标签分布
    n_edges = n_nodes * (n_nodes - 1) // 2
    n_positive = (sample_labels.triu(1) == 1).sum().item()
    n_negative = n_edges - n_positive
    
    print(f"\n3. 标签分布:")
    print(f"   正样本: {n_positive}/{n_edges} ({n_positive/n_edges*100:.1f}%)")
    print(f"   负样本: {n_negative}/{n_edges} ({n_negative/n_edges*100:.1f}%)")
    
    # 4. 总体评估
    print(f"\n4. 总体评估:")
    if silhouette >= 0.25 and separation_ratio >= 1.2:
        print(f"   ✓✓✓ 数据集质量良好，可以学习")
    elif silhouette >= 0.15 or separation_ratio >= 1.1:
        print(f"   ⚠️  数据集质量一般，可能难以学习")
    else:
        print(f"   ❌❌❌ 数据集质量差，基本无法学习")
    
    print("="*60)

# # 使用示例
# dataset = SemanticClusteringDataset(100, m=5, n=10, n_nodes=7, n_clusters=2)
# verify_dataset_quality(dataset)