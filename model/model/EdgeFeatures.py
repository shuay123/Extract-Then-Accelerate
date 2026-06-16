import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ManufacturingEdgeFeatures(nn.Module):
    """针对加工能力特征的边特征生成器"""
    
    def __init__(self, node_feature_dim, m, n):
        super().__init__()
        self.m = m
        self.n = n
        self.node_feature_dim = node_feature_dim
        
        self.product_weights = nn.Parameter(torch.ones(m))
        self.process_weights = nn.Parameter(torch.ones(n))
        
    def forward(self, node_features):
        """
        生成边特征
        node_features: [B, N, m*n]
        Returns: [B, N*(N-1), 4]
        """
        batch_size, n_nodes, _ = node_features.shape
        
        # 重塑为 [B, N, m, n]
        capabilities = node_features.reshape(batch_size, n_nodes, self.m, self.n)
        
        # 扩展以计算所有节点对
        cap_i = capabilities.unsqueeze(2)  # [B, N, 1, m, n]
        cap_j = capabilities.unsqueeze(1)  # [B, 1, N, m, n]
        
        # 应用权重
        weights = self.product_weights.view(1, 1, 1, self.m, 1) * \
                  self.process_weights.view(1, 1, 1, 1, self.n)
        
        # 1. 加权欧几里得相似性
        weighted_diff = (cap_i - cap_j) * weights
        capability_distance = torch.norm(
            weighted_diff.reshape(batch_size, n_nodes, n_nodes, -1), 
            dim=-1, keepdim=True
        )
        capability_similarity = torch.exp(-capability_distance)  # [B, N, N, 1]
        
        # 2. 能力互补性
        complementarity = torch.sum(
            torch.minimum(cap_i, cap_j) * weights, 
            dim=(-2, -1)
        ).unsqueeze(-1)
        complementarity = torch.sigmoid(complementarity)  # [B, N, N, 1]
        
        # 3. 能力重叠度
        overlap = torch.sum(
            (cap_i > 0).float() * (cap_j > 0).float() * weights,
            dim=(-2, -1)
        ).unsqueeze(-1)
        overlap = overlap / (self.m * self.n + 1e-8)  # [B, N, N, 1]
        
        # 4. 总能力差异
        total_cap_i = capabilities.sum(dim=(-2, -1))  # [B, N]
        total_cap_j = capabilities.sum(dim=(-2, -1))  # [B, N]
        
        total_cap_i = total_cap_i.unsqueeze(2)  # [B, N, 1]
        total_cap_j = total_cap_j.unsqueeze(1)  # [B, 1, N]
        
        total_cap_diff = torch.abs(total_cap_i - total_cap_j)  # [B, N, N]
        total_cap_similarity = torch.exp(-total_cap_diff).unsqueeze(-1)  # [B, N, N, 1]
        
        # 合并所有特征
        edge_features = torch.cat([
            capability_similarity,
            complementarity,
            overlap,
            total_cap_similarity,
        ], dim=-1)  # [B, N, N, 4]
        
        # 移除自环，转换为 [B, N*(N-1), 4]
        edge_features_flat = edge_features.reshape(batch_size, n_nodes * n_nodes, 4)
        mask = ~torch.eye(n_nodes, dtype=torch.bool, device=node_features.device)
        mask_flat = mask.reshape(-1)
        edge_features = edge_features_flat[:, mask_flat, :]  # [B, N*(N-1), 4]
        
        return edge_features