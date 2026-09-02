import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LabelConsistentAugmentation:
    """保持标签一致性的图增强策略"""
    
    @staticmethod
    def global_scaling(node_features, scale_range=(0.95, 1.05)):
        batch_size = node_features.shape[0]
        scale = torch.empty(batch_size, 1, 1, device=node_features.device).uniform_(*scale_range)
        return node_features * scale
    
    @staticmethod
    def uniform_noise(node_features, noise_std=0.02):
        noise = torch.randn_like(node_features) * noise_std
        augmented = node_features + noise
        augmented = torch.clamp(augmented, min=0)
        return augmented
    
    @staticmethod
    def feature_permutation(node_features, m, n):
        batch_size, n_nodes, _ = node_features.shape
        features = node_features.reshape(batch_size, n_nodes, m, n)
        
        if torch.rand(1).item() > 0.5:
            perm = torch.randperm(m)
            features = features[:, :, perm, :]
        
        if torch.rand(1).item() > 0.5:
            perm = torch.randperm(n)
            features = features[:, :, :, perm]
        
        return features.reshape(batch_size, n_nodes, -1)
    
    @staticmethod
    def capability_normalization(node_features):
        min_val = node_features.min(dim=-1, keepdim=True)[0]
        max_val = node_features.max(dim=-1, keepdim=True)[0]
        normalized = (node_features - min_val) / (max_val - min_val + 1e-8)
        
        mask = torch.rand(node_features.shape[0], 1, 1, device=node_features.device) > 0.5
        return torch.where(mask, normalized, node_features)
    
    @staticmethod
    def combined_augmentation(node_features, m, n):
        aug_methods = [
            lambda x: LabelConsistentAugmentation.global_scaling(x, scale_range=(0.95, 1.05)),
            lambda x: LabelConsistentAugmentation.uniform_noise(x, noise_std=0.02),
            lambda x: LabelConsistentAugmentation.feature_permutation(x, m, n),
            lambda x: LabelConsistentAugmentation.capability_normalization(x),
        ]
        
        n_augs = torch.randint(1, 3, (1,)).item()
        selected_augs = torch.randperm(len(aug_methods))[:n_augs]
        
        augmented = node_features
        for idx in selected_augs:
            augmented = aug_methods[idx](augmented)
        
        return augmented

class SupervisedGraphContrastiveLearning(nn.Module):
    """有监督图对比学习"""
    
    def __init__(self, hidden_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )
    
    def forward(self, node_embeddings_1, node_embeddings_2, labels):
        """
        Args:
            node_embeddings_1: [B, N, hidden_dim]
            node_embeddings_2: [B, N, hidden_dim]
            labels: [B, N, N]
        Returns:
            contrastive_loss: scalar
        """
        batch_size, n_nodes, _ = node_embeddings_1.shape
        
        z1 = self.projection_head(node_embeddings_1)  # [B, N, 128]
        z2 = self.projection_head(node_embeddings_2)  # [B, N, 128]
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        loss = 0.0
        num_valid_samples = 0
        
        for b in range(batch_size):
            z1_b = z1[b]  # [N, 128]
            z2_b = z2[b]  # [N, 128]
            labels_b = labels[b]  # [N, N]
            
            sim_matrix = torch.matmul(z1_b, z2_b.T) / self.temperature  # [N, N]
            
            for i in range(n_nodes):
                positive_mask = labels_b[i] > 0
                
                if positive_mask.sum() > 0:
                    logits = sim_matrix[i]
                    exp_logits = torch.exp(logits)
                    
                    positive_sum = (exp_logits * positive_mask.float()).sum()
                    all_sum = exp_logits.sum()
                    
                    if all_sum > 1e-8:
                        loss += -torch.log(positive_sum / all_sum + 1e-8)
                        num_valid_samples += 1
        
        if num_valid_samples > 0:
            loss = loss / num_valid_samples
        else:
            loss = torch.tensor(0.0, device=node_embeddings_1.device)
        
        return loss


class EdgeLevelContrastiveLearning_old(nn.Module):
    """
    边级对比学习
    
    核心思想：
    - 正样本：标签相同的边（都是1或都是0）
    - 负样本：标签不同的边（一个是1，一个是0）
    - 目标：让同类边的表示相似，异类边的表示不同
    """
    
    def __init__(self, hidden_dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
        # 投影头：将边嵌入投影到对比学习空间
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
        )
    
    def forward(self, edge_embeddings_1, edge_embeddings_2, edge_labels):
        """
        Args:
            edge_embeddings_1: [B, N*(N-1), hidden_dim] - 第一个视图的边嵌入
            edge_embeddings_2: [B, N*(N-1), hidden_dim] - 第二个视图的边嵌入
            edge_labels: [B, N, N] - 边标签矩阵（对称的，对角线为1）
        
        Returns:
            contrastive_loss: scalar
        """
        batch_size, n_edges, hidden_dim = edge_embeddings_1.shape
        
        # 从边数量推算节点数量：n_edges = N*(N-1)
        # N^2 - N - n_edges = 0 => N = (1 + sqrt(1 + 4*n_edges)) / 2
        n_nodes = int((1 + (1 + 4 * n_edges) ** 0.5) / 2)
        
        # 投影到对比学习空间
        z1 = self.projection_head(edge_embeddings_1)  # [B, N*(N-1), 128]
        z2 = self.projection_head(edge_embeddings_2)  # [B, N*(N-1), 128]
        
        # L2归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # 将边标签矩阵展平（去除对角线）
        edge_labels_flat = self._flatten_edge_labels(edge_labels, n_nodes)  # [B, N*(N-1)]
        
        total_loss = 0.0
        num_valid_edges = 0
        
        for b in range(batch_size):
            z1_b = z1[b]  # [N*(N-1), 128]
            z2_b = z2[b]  # [N*(N-1), 128]
            labels_b = edge_labels_flat[b]  # [N*(N-1)]
            
            # 计算边之间的相似度矩阵
            sim_matrix = torch.matmul(z1_b, z2_b.T) / self.temperature  # [N*(N-1), N*(N-1)]
            
            # 对每条边，找到同类边作为正样本
            for i in range(n_edges):
                # 正样本：标签相同的其他边
                # 如果边i的标签是1，那么所有标签为1的其他边都是正样本
                # 如果边i的标签是0，那么所有标签为0的其他边都是正样本
                positive_mask = (labels_b == labels_b[i]) & (torch.arange(n_edges, device=labels_b.device) != i)
                
                # 如果没有正样本，跳过这条边
                if positive_mask.sum() == 0:
                    continue
                
                logits = sim_matrix[i]  # [N*(N-1)]
                exp_logits = torch.exp(logits)
                
                # 正样本的exp之和
                positive_sum = (exp_logits * positive_mask.float()).sum()
                
                # 所有样本的exp之和（包括负样本）
                all_sum = exp_logits.sum()
                
                # 对比学习损失
                loss = -torch.log(positive_sum / all_sum + 1e-8)
                total_loss += loss
                num_valid_edges += 1
        
        if num_valid_edges > 0:
            avg_loss = total_loss / num_valid_edges
        else:
            avg_loss = torch.tensor(0.0, device=edge_embeddings_1.device)
        
        return avg_loss
    
    def _flatten_edge_labels(self, edge_labels, n_nodes):
        """
        将边标签矩阵 [B, N, N] 展平为 [B, N*(N-1)]
        去除对角线元素
        
        Args:
            edge_labels: [B, N, N]
            n_nodes: int
        
        Returns:
            labels_flat: [B, N*(N-1)]
        """
        batch_size = edge_labels.shape[0]
        device = edge_labels.device
        
        # 创建非对角线mask
        mask = ~torch.eye(n_nodes, dtype=torch.bool, device=device)
        
        # 对每个batch，提取非对角线元素
        labels_flat = []
        for b in range(batch_size):
            labels_b = edge_labels[b]  # [N, N]
            labels_b_flat = labels_b[mask]  # [N*(N-1)]
            labels_flat.append(labels_b_flat)
        
        labels_flat = torch.stack(labels_flat, dim=0)  # [B, N*(N-1)]
        return labels_flat
    
    import torch

class EdgeLevelContrastiveLearning_v1(nn.Module):
    """
    边级监督对比学习（向量化加速版）

    默认：same_label 将同标签边作为正样本。
    可选：pos_label_only=1 只把 label==1 当正样本（更推荐，避免0类主导表示）
    """
    def __init__(self, hidden_dim, temperature=0.5, pos_label_only=None):
        super().__init__()
        self.temperature = temperature
        self.pos_label_only = pos_label_only  # None: same_label; 1: only label==1 as positive

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
        )

    @staticmethod
    def _upper_tri_mask(n, device):
        # 上三角（不含对角线）
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _directed_edges_to_undirected_upper(z, n_nodes):
        """
        将有向边序列 [B, N*(N-1), C] 还原到 [B,N,N,C]（对角线为0），
        再对称化，最后取无向上三角边 [B, Eu, C]。
        """
        B, E, C = z.shape
        device = z.device

        # [N*N] 的非对角mask，与 model.py 的重构方式一致
        no_diag = ~torch.eye(n_nodes, device=device, dtype=torch.bool)
        no_diag_flat = no_diag.view(-1)

        full = torch.zeros(B, n_nodes * n_nodes, C, device=device, dtype=z.dtype)
        full[:, no_diag_flat, :] = z
        full = full.view(B, n_nodes, n_nodes, C)

        # 无向化
        und = (full + full.transpose(1, 2)) * 0.5

        tri = EdgeLevelContrastiveLearning._upper_tri_mask(n_nodes, device)
        z_u = und[:, tri, :]  # [B, Eu, C]
        return z_u, tri

    def forward(self, edge_embeddings_1, edge_embeddings_2, edge_labels):
        """
        edge_embeddings_*: [B, N*(N-1), hidden_dim]（有向边序列）
        edge_labels:       [B, N, N]（对称标签矩阵，含对角线）
        """
        B, n_edges, _ = edge_embeddings_1.shape
        # n_edges = N*(N-1) => N = (1 + sqrt(1+4E))/2
        n_nodes = int((1 + (1 + 4 * n_edges) ** 0.5) / 2)

        # projection + normalize
        z1 = F.normalize(self.projection_head(edge_embeddings_1), dim=-1)  # [B,E,128]
        z2 = F.normalize(self.projection_head(edge_embeddings_2), dim=-1)  # [B,E,128]

        # 转无向上三角边（Eu = N*(N-1)/2）
        z1_u, tri = self._directed_edges_to_undirected_upper(z1, n_nodes)  # [B,Eu,128]
        z2_u, _   = self._directed_edges_to_undirected_upper(z2, n_nodes)  # [B,Eu,128]

        # 对应的无向上三角标签 [B,Eu]
        labels_u = edge_labels[:, tri].long()

        # logits: [B,Eu,Eu]
        logits = torch.bmm(z1_u, z2_u.transpose(1, 2)) / self.temperature

        Eu = labels_u.size(1)
        eye = torch.eye(Eu, device=logits.device, dtype=torch.bool).unsqueeze(0)  # [1,Eu,Eu]

        if self.pos_label_only is None:
            # Treat edges with the same label as positive pairs, including 0-0 and 1-1.
            pos_mask = labels_u.unsqueeze(2).eq(labels_u.unsqueeze(1)) & (~eye)
        else:
            # Restrict positive pairs to edges whose label equals pos_label_only.
            pos_mask = (labels_u == self.pos_label_only)
            pos_mask = (pos_mask.unsqueeze(2) & pos_mask.unsqueeze(1)) & (~eye)

        # log denom: log sum exp over all j
        log_denom = torch.logsumexp(logits, dim=2)  # [B,Eu]

        # log num: log sum exp over positives
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(logits_pos, dim=2)  # [B,Eu]

        valid = torch.isfinite(log_num)  # 没有正样本时为 -inf
        if valid.any():
            loss = -(log_num[valid] - log_denom[valid]).mean()
        else:
            loss = logits.sum() * 0.0  # 保持可反传、避免nan

        return loss


class EdgeLevelContrastiveLearning(nn.Module):
    """
    有向边级监督对比学习（向量化 + 类均衡采样）

    - 不做无向上三角：保留 e_ij 与 e_ji 的差异
    - 类均衡采样：每个样本内，从 0/1 类各采 sample_per_class 条（不足则取尽量多）
    - pos_label_only:
        None: same_label 为正样本（0-0 & 1-1 都算正）
         1  : 仅 label==1 的边互为正样本（更常用于“边存在”为正类的任务）
    """
    def __init__(
        self,
        hidden_dim,
        temperature=0.5,
        pos_label_only=None,
        sample_per_class=256,    # Typical range: 128-512; None uses all edges.
        symmetric=False          # 可选：是否做 z1->z2 与 z2->z1 双向平均
    ):
        super().__init__()
        self.temperature = temperature
        self.pos_label_only = pos_label_only
        self.sample_per_class = sample_per_class
        self.symmetric = symmetric

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
        )

    @staticmethod
    def _labels_to_directed_flat(edge_labels: torch.Tensor) -> torch.Tensor:
        """
        edge_labels: [B, N, N]
        return:      [B, N*(N-1)]  (row-major，去掉对角线，顺序与 EdgeFeatures 的 flatten+mask 对齐)
        """
        B, N, _ = edge_labels.shape
        device = edge_labels.device
        mask = ~torch.eye(N, device=device, dtype=torch.bool)  # [N,N]
        flat = edge_labels.reshape(B, N * N)[:, mask.reshape(-1)]  # [B, N*(N-1)]
        return flat

    @staticmethod
    def _sample_k(idx: torch.Tensor, k: int) -> torch.Tensor:
        """在 idx 中随机采 k 个（不放回）；idx 在 GPU 上也可用。"""
        if idx.numel() <= k:
            return idx
        perm = torch.randperm(idx.numel(), device=idx.device)[:k]
        return idx[perm]

    def _balanced_indices(self, labels_1d: torch.Tensor) -> torch.Tensor:
        """
        labels_1d: [E] (0/1)
        return:    [M] 采样后的边索引（尽量 0/1 各 sample_per_class 条）
        """
        E = labels_1d.numel()
        if self.sample_per_class is None:
            return torch.arange(E, device=labels_1d.device)

        idx0 = torch.nonzero(labels_1d == 0, as_tuple=False).squeeze(1)
        idx1 = torch.nonzero(labels_1d == 1, as_tuple=False).squeeze(1)

        k0 = min(self.sample_per_class, idx0.numel())
        k1 = min(self.sample_per_class, idx1.numel())

        # Use the smaller class size to keep the two classes balanced.
        k = min(k0, k1)
        if k == 0:
            # 某一类缺失：退化为随机采一些（否则没法构造正样本）
            k_any = min(self.sample_per_class, E)
            idx_all = torch.arange(E, device=labels_1d.device)
            return self._sample_k(idx_all, k_any)

        s0 = self._sample_k(idx0, k)
        s1 = self._sample_k(idx1, k)
        return torch.cat([s0, s1], dim=0)  # [2k]

    def _contrast_one(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        z1,z2: [M, C]  (已 normalize)
        y:     [M]     (0/1)
        return: scalar loss（可能为 0）
        """
        M = y.numel()
        if M < 2:
            return z1.sum() * 0.0

        logits = (z1 @ z2.t()) / self.temperature  # [M,M]
        eye = torch.eye(M, device=logits.device, dtype=torch.bool)

        if self.pos_label_only is None:
            # same label 为正样本
            pos_mask = y.view(M, 1).eq(y.view(1, M)) & (~eye)
            anchor_mask = torch.ones(M, device=logits.device, dtype=torch.bool)
        else:
            # 仅 label==pos_label_only 的边参与正样本配对
            pos_nodes = (y == self.pos_label_only)
            pos_mask = (pos_nodes.view(M, 1) & pos_nodes.view(1, M)) & (~eye)
            anchor_mask = pos_nodes

        log_denom = torch.logsumexp(logits, dim=1)  # [M]
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(logits_pos, dim=1)  # [M]

        valid = torch.isfinite(log_num) & anchor_mask  # 至少有一个正样本
        if valid.any():
            return -(log_num[valid] - log_denom[valid]).mean()
        else:
            return logits.sum() * 0.0

    def forward(self, edge_embeddings_1, edge_embeddings_2, edge_labels):
        """
        edge_embeddings_*: [B, N*(N-1), hidden_dim]（有向边序列）
        edge_labels:       [B, N, N]（通常对称，但这里按有向边 flatten；e_ij 与 e_ji 可不同）
        """
        # projection + normalize
        z1 = F.normalize(self.projection_head(edge_embeddings_1), dim=-1)  # [B,E,128]
        z2 = F.normalize(self.projection_head(edge_embeddings_2), dim=-1)  # [B,E,128]

        # labels -> directed flat [B,E]
        y = self._labels_to_directed_flat(edge_labels).long()

        B, E, _ = z1.shape
        losses = []

        # Iterate only over batches; B is typically between 8 and 64.
        for b in range(B):
            idx = self._balanced_indices(y[b])          # [M]
            z1b = z1[b, idx, :]                         # [M,C]
            z2b = z2[b, idx, :]                         # [M,C]
            yb  = y[b, idx]                             # [M]

            loss_b = self._contrast_one(z1b, z2b, yb)
            if self.symmetric:
                loss_b = 0.5 * (loss_b + self._contrast_one(z2b, z1b, yb))

            losses.append(loss_b)

        return torch.stack(losses).mean()
