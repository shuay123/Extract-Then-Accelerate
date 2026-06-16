import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.EdgeFeatures import ManufacturingEdgeFeatures
from model.layer import SparseGCNLayer, BatchNormNode
from model.ContrastiveLearning import SupervisedGraphContrastiveLearning,LabelConsistentAugmentation,EdgeLevelContrastiveLearning

class UndirectedContrastiveClusteringModel(nn.Module):
    """
    无向边 + 对比学习 + 论文边损失
    
    关键设计：
    1. 边嵌入：仍然使用N*(N-1)条有向边进行GCN消息传递
    2. 边预测：对每个节点的出边做softmax（论文方法）
    3. 损失计算：只对上三角边计算损失（避免重复）
    4. 推理输出：对称的邻接矩阵
    """
    
    def __init__(self, m, n, n_nodes=7, hidden_dim=64, n_gcn_layers=4, 
                 temperature=0.07, use_contrastive=True):
        super().__init__()
        self.m = m
        self.n = n
        self.n_nodes = n_nodes
        self.node_feature_dim = m * n
        self.use_contrastive = use_contrastive
        
        # 边特征生成器
        self.edge_feature_generator = ManufacturingEdgeFeatures(
            self.node_feature_dim, m, n
        )
        edge_feature_dim = 4
        
        # 节点和边嵌入
        self.nodes_embedding = nn.Linear(self.node_feature_dim, hidden_dim)
        self.edges_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList([
            SparseGCNLayer(hidden_dim) 
            for _ in range(n_gcn_layers)
        ])
        
        # 边分类器（输出logits）
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 对比学习模块
        if use_contrastive:
            self.contrastive_module = EdgeLevelContrastiveLearning(
                hidden_dim, temperature
            )
        
        self.augmentation = LabelConsistentAugmentation()
        self.edge_class_weights = None
    
    def get_upper_triangular_mask(self, n_nodes, device):
        """
        获取上三角mask（不包括对角线）
        Returns: [N*N] bool tensor
        """
        mask_2d = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1)
        mask_flat = mask_2d.reshape(-1).bool()
        return mask_flat
    
    def encode(self, node_features):
        """
        编码节点和边
        
        Returns:
            x: [B, N, hidden_dim] - 节点嵌入
            e: [B, N*(N-1), hidden_dim] - 边嵌入（有向）
        """
        batch_size, n_nodes, _ = node_features.shape
        
        # 生成边特征: [B, N*(N-1), 4]
        edge_features = self.edge_feature_generator(node_features)
        
        # 初始嵌入
        x = self.nodes_embedding(node_features)      # [B, N, hidden_dim]
        e = self.edges_embedding(edge_features)      # [B, N*(N-1), hidden_dim]
        
        # 构建edge_index（全连接有向图）
        edge_index = self.build_edge_index(n_nodes, batch_size, node_features.device)
        inverse_edge_index = self.build_inverse_edge_index(
            n_nodes, batch_size, node_features.device
        )
        
        # 图卷积
        for layer in self.gcn_layers:
            x, e = layer(x, e, edge_index, inverse_edge_index, n_nodes - 1)
        # x: [B, N, hidden_dim], e: [B, N*(N-1), hidden_dim]
        
        return x, e
    
    def forward(self, node_features, labels=None, epoch=0):
        """
        前向传播
        
        Args:
            node_features: [B, N, m*n]
            labels: [B, N, N] - 对称的邻接矩阵
            epoch: 当前epoch
        
        Returns:
            edge_scores: [B, N, N] - 对称的预测邻接矩阵
            loss_dict: 损失字典
        """
        batch_size, n_nodes, _ = node_features.shape
        # assert n_nodes == self.n_nodes
        
        # 对比学习：生成两个增强视图
        if self.use_contrastive and self.training and labels is not None:
            node_features_aug1 = self.augmentation.combined_augmentation(
                node_features, self.m, self.n
            )
            node_features_aug2 = self.augmentation.combined_augmentation(
                node_features, self.m, self.n
            )
            
            node_emb_1, edge_emb_1 = self.encode(node_features_aug1)
            node_emb_2, edge_emb_2 = self.encode(node_features_aug2)
            
            edge_emb = (edge_emb_1 + edge_emb_2) / 2  # [B, N*(N-1), hidden_dim]
        else:
            node_emb_1, edge_emb = self.encode(node_features)
            node_emb_2 = None
        
        # ============ 边预测（论文方法 + 无向边）============
        # Step 1: 获取原始logits
        edge_logits_flat = self.edge_classifier(edge_emb).squeeze(-1)
        # edge_logits_flat: [B, N*(N-1)]
        
        # Step 2: 重塑为 [B, N, N-1]，每个节点有N-1条出边
        edge_logits = edge_logits_flat.view(batch_size, n_nodes, n_nodes - 1)
        # edge_logits: [B, N, N-1]
        
        # Step 3: 对每个节点的出边做sigmoid（二分类）
        edge_probs = F.sigmoid(edge_logits)
        # edge_probs: [B, N, N-1] - 每个节点的出边概率分布
        
        # Step 4: 恢复为完整邻接矩阵（不含对角线）
        edge_probs_flat = edge_probs.view(batch_size, n_nodes * (n_nodes - 1))
        # edge_probs_flat: [B, N*(N-1)]
        
        # 重构为 [B, N, N]
        edge_scores_full = torch.zeros(
            batch_size, n_nodes * n_nodes, device=node_features.device
        )
        no_diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=node_features.device)
        no_diag_mask_flat = no_diag_mask.reshape(-1)
        edge_scores_full[:, no_diag_mask_flat] = edge_probs_flat
        edge_scores_full = edge_scores_full.reshape(batch_size, n_nodes, n_nodes)
        # edge_scores_full: [B, N, N] - 有向边概率（对角线为0）
        
        # Step 5: 对称化得到无向边得分
        edge_scores = (edge_scores_full + edge_scores_full.transpose(1, 2)) / 2
        # edge_scores: [B, N, N] - 对称矩阵
        
        # 对角线设为1
        edge_scores[:, range(n_nodes), range(n_nodes)] = 1.0
        
        # ============ 损失计算 ============
        loss_dict = {}
        if labels is not None:
            # 论文方法的边损失（只对上三角边计算，避免重复）
            
            # 获取上三角mask
            upper_tri_mask = self.get_upper_triangular_mask(n_nodes, labels.device)
            # upper_tri_mask: [N*N] - 上三角位置为True
            
            # 准备标签和预测
            labels_flat = labels.reshape(batch_size, -1)[:, upper_tri_mask].long()
            # labels_flat: [B, N*(N-1)/2]
            
            # 对有向边的预测做对称化后再提取上三角
            # 使用edge_scores而不是edge_scores_full，因为已经对称化了
            edge_scores_upper = edge_scores.reshape(batch_size, -1)[:, upper_tri_mask]
            # edge_scores_upper: [B, N*(N-1)/2]
            
            # 构造两类log概率 [log(1-p), log(p)]
            edge_log_probs = torch.stack([
                torch.log(1 - edge_scores_upper + 1e-8),
                torch.log(edge_scores_upper + 1e-8)
            ], dim=-1)  # [B, N*(N-1)/2, 2]
            
            # 计算类别权重（只在第一次）
            if self.edge_class_weights is None and self.training:
                labels_np = labels_flat.cpu().numpy().flatten()
                class_counts = np.bincount(labels_np)
                weights = len(labels_np) / (len(class_counts) * class_counts + 1e-8)
                self.edge_class_weights = torch.tensor(
                    weights, dtype=torch.float32, device=node_features.device
                )
                print(f"✓ 边类别权重（无向边）: 类0={weights[0]:.3f}, 类1={weights[1]:.3f}")
                print(f"✓ 边数量: {labels_flat.shape[1]} (上三角无向边)")
            
            # NLLLoss
            edge_log_probs = edge_log_probs.permute(0, 2, 1)  # [B, 2, N*(N-1)/2]
            
            if self.edge_class_weights is not None:
                loss_edge = F.nll_loss(
                    edge_log_probs, labels_flat, weight=self.edge_class_weights
                )
            else:
                loss_edge = F.nll_loss(edge_log_probs, labels_flat)
            
            loss_dict['loss_edge'] = loss_edge
            
            # 对比学习损失
            if self.use_contrastive and self.training and labels is not None:
                loss_contrast = self.contrastive_module(
                    edge_emb_1, edge_emb_2, labels
                )
                loss_dict['loss_contrast'] = loss_contrast
                
                # 动态权重
                alpha = max(0.3, 1.0 - epoch / 100)
                beta = 1.0
                
                total_loss = alpha * loss_contrast + beta * loss_edge
                loss_dict['total_loss'] = total_loss
            else:
                loss_dict['total_loss'] = loss_edge
        
        return edge_scores, loss_dict
    
    def build_edge_index(self, n_nodes, batch_size, device):
        """构建全连接图的edge_index（用于GCN）"""
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_index.append(j)
        edge_index = torch.tensor(edge_index, device=device, dtype=torch.long)
        return edge_index.unsqueeze(0).expand(batch_size, -1)
    
    def build_inverse_edge_index(self, n_nodes, batch_size, device):
        """构建反向边索引"""
        inverse_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    reverse_edge_idx = j * (n_nodes - 1) + (i if i < j else i - 1)
                    inverse_index.append(reverse_edge_idx)
        inverse_index = torch.tensor(inverse_index, device=device, dtype=torch.long)
        return inverse_index.unsqueeze(0).expand(batch_size, -1)