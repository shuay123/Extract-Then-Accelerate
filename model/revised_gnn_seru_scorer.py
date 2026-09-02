import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== 第一部分：基础层 ====================

class BatchNormNode(nn.Module):
    """节点批归一化"""
    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        # x: [batch_size, n_nodes, hidden_dim]
        x_trans = x.transpose(1, 2).contiguous()
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()
        return x_bn


class NodeFeatures(nn.Module):
    """节点特征更新模块"""
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.node_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.to_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.edge_embedding = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e, edge_index, n_edges):
        batch_size, num_nodes, hidden_dim = x.size()
        
        Ux = self.node_embedding(x)
        Vx = self.to_embedding(x)
        Ve = self.edge_embedding(e)
        
        # 边注意力
        Ve = F.softmax(Ve.view(batch_size, num_nodes, n_edges, hidden_dim), dim=2)
        Ve = Ve.view(batch_size, num_nodes * n_edges, hidden_dim)
        
        # 获取目标节点特征
        Vx = Vx[torch.arange(batch_size).view(-1, 1), edge_index]
        
        # 加权聚合
        to = Ve * Vx
        to = to.view(batch_size, num_nodes, n_edges, hidden_dim).sum(2)
        
        x_new = Ux + to
        return x_new


class EdgeFeatures(nn.Module):
    """边特征更新模块。"""
    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.hidden_dim = hidden_dim  # 保存hidden_dim
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_from = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_to = nn.Linear(hidden_dim, hidden_dim, True)
        self.inverse_U = nn.Linear(hidden_dim, hidden_dim, True)
        self.W_placeholder = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_placeholder.data.uniform_(-1, 1)

    def forward(self, x, e, edge_index, inverse_edge_index, n_edges):
        batch_size, graph_size, hidden_dim = x.size()
        
        Ue = self.U(e)
        inverse_Ue = self.inverse_U(e)
        
        # 添加占位符
        inverse_Ue = torch.cat(
            (inverse_Ue, self.W_placeholder.view(1, 1, hidden_dim).repeat(batch_size, 1, 1)), 
            1
        )
        inverse_node_embedding = inverse_Ue[
            torch.arange(batch_size).view(batch_size, 1), 
            inverse_edge_index
        ]
        
        Vx_from = self.V_from(x)
        Vx_to = self.V_to(x)
        Vx = Vx_to[torch.arange(batch_size).view(-1, 1), edge_index]
        
        # Use the configured hidden dimension instead of a fixed width.
        Vx = Vx.view(batch_size, -1, n_edges, self.hidden_dim) + \
             Vx_from.view(batch_size, -1, 1, self.hidden_dim)
        Vx = Vx.view(batch_size, -1, self.hidden_dim)
        
        e_new = Ue + Vx + inverse_node_embedding
        return e_new


class SparseGCNLayer(nn.Module):
    """稀疏图卷积层"""
    def __init__(self, hidden_dim, aggregation="mean"):
        super(SparseGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormNode(hidden_dim)

    def forward(self, x, e, edge_index, inverse_edge_index, n_edges):
        e_in = e
        x_in = x

        # 更新节点
        x_tmp = self.node_feat(x_in, e_in, edge_index.long(), n_edges)
        x_tmp = self.bn_node(x_tmp)
        x = F.relu(x_tmp)
        x_new = x_in + x

        # 更新边
        e_tmp = self.edge_feat(x_new, e_in, edge_index.long(), 
                               inverse_edge_index.long(), n_edges)
        e_tmp = self.bn_edge(e_tmp)
        e = F.relu(e_tmp)
        e_new = e_in + e
        
        return x_new, e_new


# ==================== 第二部分：边特征生成器 ====================

class ManufacturingEdgeFeatures(nn.Module):
    """针对加工能力特征的边特征生成器。"""
    
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
        Returns: [B, N*(N-1), edge_dim]
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
        capability_similarity = torch.exp(-capability_distance)
        # Shape: [B, N, N, 1]
        
        # 2. 能力互补性
        complementarity = torch.sum(
            torch.minimum(cap_i, cap_j) * weights, 
            dim=(-2, -1)
        ).unsqueeze(-1)
        complementarity = torch.sigmoid(complementarity)
        # Shape: [B, N, N, 1]
        
        # 3. 能力重叠度
        overlap = torch.sum(
            (cap_i > 0).float() * (cap_j > 0).float() * weights,
            dim=(-2, -1)
        ).unsqueeze(-1)
        overlap = overlap / (self.m * self.n + 1e-8)
        # Shape: [B, N, N, 1]
        
        # 4. Total capability difference without keepdim.
        total_cap_i = capabilities.sum(dim=(-2, -1))  # [B, N]
        total_cap_j = capabilities.sum(dim=(-2, -1))  # [B, N]
        
        total_cap_i = total_cap_i.unsqueeze(2)  # [B, N, 1]
        total_cap_j = total_cap_j.unsqueeze(1)  # [B, 1, N]
        
        total_cap_diff = torch.abs(total_cap_i - total_cap_j)  # [B, N, N]
        total_cap_similarity = torch.exp(-total_cap_diff).unsqueeze(-1)  # [B, N, N, 1]
        # Shape: [B, N, N, 1]
        
        # Validate tensor dimensions.
        assert capability_similarity.shape == (batch_size, n_nodes, n_nodes, 1)
        assert complementarity.shape == (batch_size, n_nodes, n_nodes, 1)
        assert overlap.shape == (batch_size, n_nodes, n_nodes, 1)
        assert total_cap_similarity.shape == (batch_size, n_nodes, n_nodes, 1)
        
        # 合并所有特征
        edge_features = torch.cat([
            capability_similarity,
            complementarity,
            overlap,
            total_cap_similarity,
        ], dim=-1)  # [B, N, N, 4]
        
        # 移除自环
        edge_features_flat = edge_features.reshape(batch_size, n_nodes * n_nodes, 4)
        mask = ~torch.eye(n_nodes, dtype=torch.bool, device=node_features.device)
        mask_flat = mask.reshape(-1)
        edge_features = edge_features_flat[:, mask_flat, :]  # [B, N*(N-1), 4]
        
        return edge_features


# ==================== 第三部分：保持标签一致性的增强策略 ====================

class LabelConsistentAugmentation:
    """保持标签一致性的图增强策略"""
    
    @staticmethod
    def global_scaling(node_features, scale_range=(0.95, 1.05)):
        """全局等比例缩放"""
        batch_size = node_features.shape[0]
        scale = torch.empty(batch_size, 1, 1, device=node_features.device).uniform_(*scale_range)
        return node_features * scale
    
    @staticmethod
    def uniform_noise(node_features, noise_std=0.02):
        """添加非常小的均匀噪声"""
        noise = torch.randn_like(node_features) * noise_std
        augmented = node_features + noise
        augmented = torch.clamp(augmented, min=0)
        return augmented
    
    @staticmethod
    def feature_permutation(node_features, m, n):
        """交换产品或工序的顺序"""
        batch_size, n_nodes, _ = node_features.shape
        features = node_features.reshape(batch_size, n_nodes, m, n)
        
        # 随机打乱产品维度
        if torch.rand(1).item() > 0.5:
            perm = torch.randperm(m)
            features = features[:, :, perm, :]
        
        # 随机打乱工序维度
        if torch.rand(1).item() > 0.5:
            perm = torch.randperm(n)
            features = features[:, :, :, perm]
        
        return features.reshape(batch_size, n_nodes, -1)
    
    @staticmethod
    def capability_normalization(node_features):
        """归一化变换"""
        # 按节点维度归一化
        min_val = node_features.min(dim=-1, keepdim=True)[0]
        max_val = node_features.max(dim=-1, keepdim=True)[0]
        normalized = (node_features - min_val) / (max_val - min_val + 1e-8)
        
        # 随机选择是否归一化
        mask = torch.rand(node_features.shape[0], 1, 1, device=node_features.device) > 0.5
        return torch.where(mask, normalized, node_features)
    
    @staticmethod
    def combined_augmentation(node_features, m, n):
        """组合多种增强"""
        aug_methods = [
            lambda x: LabelConsistentAugmentation.global_scaling(x, scale_range=(0.95, 1.05)),
            lambda x: LabelConsistentAugmentation.uniform_noise(x, noise_std=0.02),
            lambda x: LabelConsistentAugmentation.feature_permutation(x, m, n),
            lambda x: LabelConsistentAugmentation.capability_normalization(x),
        ]
        
        # 随机应用1-2个增强
        n_augs = torch.randint(1, 3, (1,)).item()
        selected_augs = torch.randperm(len(aug_methods))[:n_augs]
        
        augmented = node_features
        for idx in selected_augs:
            augmented = aug_methods[idx](augmented)
        
        return augmented


# ==================== 第四部分：有监督图对比学习 ====================

class SupervisedGraphContrastiveLearning(nn.Module):
    """有监督图对比学习。"""
    
    def __init__(self, hidden_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        # 投影头
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
            labels: [B, N, N] - 边标签
        Returns:
            contrastive_loss: scalar
        """
        batch_size, n_nodes, _ = node_embeddings_1.shape
        
        # 投影到对比学习空间
        z1 = self.projection_head(node_embeddings_1)
        z2 = self.projection_head(node_embeddings_2)
        
        # L2归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        loss = 0.0
        num_valid_samples = 0  # 记录有效样本数
        
        for b in range(batch_size):
            z1_b = z1[b]  # [N, 128]
            z2_b = z2[b]  # [N, 128]
            labels_b = labels[b]  # [N, N]
            
            # 计算相似度矩阵
            sim_matrix = torch.matmul(z1_b, z2_b.T) / self.temperature
            
            for i in range(n_nodes):
                positive_mask = labels_b[i] > 0
                
                # Skip anchors without positive samples.
                if positive_mask.sum() > 0:
                    logits = sim_matrix[i]
                    exp_logits = torch.exp(logits)
                    
                    positive_sum = (exp_logits * positive_mask.float()).sum()
                    all_sum = exp_logits.sum()
                    
                    # Guard against non-finite loss values.
                    if all_sum > 1e-8:
                        loss += -torch.log(positive_sum / all_sum + 1e-8)
                        num_valid_samples += 1
        
        # Avoid division by zero.
        if num_valid_samples > 0:
            loss = loss / num_valid_samples
        else:
            loss = torch.tensor(0.0, device=node_embeddings_1.device)
        
        return loss


# ==================== 第五部分：完整模型 ====================

class ContrastiveClusteringModel(nn.Module):
    """结合对比学习的集合划分模型"""
    
    def __init__(self, m, n, hidden_dim=64, n_gcn_layers=4, 
                 temperature=0.07, use_contrastive=True):
        super().__init__()
        self.m = m
        self.n = n
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
        
        # 边分类器
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 对比学习模块
        if use_contrastive:
            self.contrastive_module = SupervisedGraphContrastiveLearning(
                hidden_dim, temperature
            )
        
        # 图增强
        self.augmentation = LabelConsistentAugmentation()
    
    def encode(self, node_features):
        """编码节点和边"""
        batch_size, n_nodes, _ = node_features.shape
        
        # 生成边特征
        edge_features = self.edge_feature_generator(node_features)
        
        # 初始嵌入
        x = self.nodes_embedding(node_features)
        e = self.edges_embedding(edge_features)
        
        # 构建edge_index（全连接）
        edge_index = self.build_edge_index(n_nodes, batch_size, node_features.device)
        inverse_edge_index = self.build_inverse_edge_index(
            n_nodes, batch_size, node_features.device
        )
        
        # 图卷积
        for layer in self.gcn_layers:
            x, e = layer(x, e, edge_index, inverse_edge_index, n_nodes - 1)
        
        return x, e
    
    def forward(self, node_features, labels=None, epoch=0):
        """
        Args:
            node_features: [B, N, m*n]
            labels: [B, N, N]
            epoch: 当前epoch
        """
        batch_size, n_nodes, _ = node_features.shape
        
        # 如果使用对比学习，生成两个增强视图
        if self.use_contrastive and self.training and labels is not None:
            node_features_aug1 = self.augmentation.combined_augmentation(
                node_features, self.m, self.n
            )
            node_features_aug2 = self.augmentation.combined_augmentation(
                node_features, self.m, self.n
            )
            
            # 编码两个视图
            node_emb_1, edge_emb_1 = self.encode(node_features_aug1)
            node_emb_2, edge_emb_2 = self.encode(node_features_aug2)
            
            # 平均两个视图的边嵌入
            edge_emb = (edge_emb_1 + edge_emb_2) / 2
        else:
            node_emb_1, edge_emb = self.encode(node_features)
            node_emb_2 = None
        
        # 边分类
        edge_scores_flat = torch.sigmoid(self.edge_classifier(edge_emb).squeeze(-1))
        
        # 重构为邻接矩阵
        edge_scores = torch.zeros(
            batch_size, n_nodes, n_nodes, device=node_features.device
        )
        mask = ~torch.eye(n_nodes, dtype=torch.bool, device=node_features.device)
        
        # Populate edge_scores from the flattened predictions.
        # edge_scores_flat的形状是[B, N*(N-1)]
        # 需要正确地映射回[B, N, N]
        edge_scores = edge_scores.reshape(batch_size, -1)  # [B, N*N]
        mask_flat = mask.reshape(-1)  # [N*N]
        edge_scores[:, mask_flat] = edge_scores_flat
        edge_scores = edge_scores.reshape(batch_size, n_nodes, n_nodes)
        
        # 对称化
        edge_scores = (edge_scores + edge_scores.transpose(1, 2)) / 2
        edge_scores[:, range(n_nodes), range(n_nodes)] = 1.0
        
        # 计算损失
        loss_dict = {}
        if labels is not None:
            # 边分类损失
            loss_edge = F.binary_cross_entropy(edge_scores, labels)
            loss_dict['loss_edge'] = loss_edge
            
            # 对比学习损失
            if self.use_contrastive and self.training and node_emb_2 is not None:
                loss_contrast = self.contrastive_module(
                    node_emb_1, node_emb_2, labels
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
        """构建全连接图的edge_index"""
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
                    # 边(i,j)的反向边是(j,i)
                    # (j,i)在边列表中的位置
                    reverse_edge_idx = j * (n_nodes - 1) + (i if i < j else i - 1)
                    inverse_index.append(reverse_edge_idx)
        
        # 添加一个额外的占位符索引（用于没有反向边的情况）
        inverse_index = torch.tensor(inverse_index, device=device, dtype=torch.long)
        return inverse_index.unsqueeze(0).expand(batch_size, -1)


# ==================== 第六部分：数据生成 ====================

class ClusteringDataset(Dataset):
    """集合划分数据集"""
    
    def __init__(self, n_samples, m, n, n_nodes=7, n_clusters=3):
        self.n_samples = n_samples
        self.m = m
        self.n = n
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        
        # 生成数据
        self.data = self.generate_data()
    
    def generate_data(self):
        """生成模拟数据"""
        data = []
        
        for _ in range(self.n_samples):
            # 生成节点特征（加工能力）
            node_features = torch.rand(self.n_nodes, self.m * self.n) * 0.8 + 0.1
            
            # 生成集合划分（标签）
            cluster_assignment = torch.randint(0, self.n_clusters, (self.n_nodes,))
            
            # 生成边标签
            labels = torch.zeros(self.n_nodes, self.n_nodes)
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if cluster_assignment[i] == cluster_assignment[j]:
                        labels[i, j] = 1.0
            
            data.append((node_features, labels))
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


# ==================== 第七部分：训练和评估 ====================

def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    epoch_losses = {'total': [], 'edge': [], 'contrast': []}
    
    for node_features, labels in tqdm(dataloader, desc=f'Epoch {epoch}', leave=False):
        node_features = node_features.to(device)
        labels = labels.to(device)
        
        # 前向传播
        edge_scores, loss_dict = model(node_features, labels, epoch)
        
        # 反向传播
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 记录损失
        epoch_losses['total'].append(loss_dict['total_loss'].item())
        epoch_losses['edge'].append(loss_dict['loss_edge'].item())
        if 'loss_contrast' in loss_dict:
            epoch_losses['contrast'].append(loss_dict['loss_contrast'].item())
    
    # 计算平均损失
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    return avg_losses


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for node_features, labels in dataloader:
            node_features = node_features.to(device)
            labels = labels.to(device)
            
            edge_scores, _ = model(node_features)
            
            # 转换为0/1预测
            preds = (edge_scores > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    accuracy = (all_preds == all_labels).float().mean().item()
    
    # 计算F1 score
    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def plot_training_curves(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 总损失
    axes[0, 0].plot(history['train_total_loss'], label='Train', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 边分类损失
    axes[0, 1].plot(history['train_edge_loss'], label='Train', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Edge Loss', fontsize=12)
    axes[0, 1].set_title('Edge Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 对比学习损失
    if history['train_contrast_loss']:
        axes[1, 0].plot(history['train_contrast_loss'], label='Train', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Contrastive Loss', fontsize=12)
        axes[1, 0].set_title('Contrastive Learning Loss', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Contrastive Loss', 
                       ha='center', va='center', fontsize=14)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
    
    # 准确率和F1
    epochs_val = [i * 5 for i in range(len(history['val_accuracy']))]
    axes[1, 1].plot(epochs_val, history['val_accuracy'], 
                   label='Accuracy', marker='o', linewidth=2)
    axes[1, 1].plot(epochs_val, history['val_f1'], 
                   label='F1 Score', marker='s', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("训练曲线已保存为 training_curves.png")


# ==================== 第八部分：主程序 ====================

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 配置
    config = {
        'm': 5,
        'n': 10,
        'n_nodes': 7,
        'n_clusters': 3,
        'hidden_dim': 64,
        'n_gcn_layers': 4,
        'temperature': 0.07,
        'use_contrastive': True,
        'batch_size': 1,
        'n_epochs': 100,
        'lr': 0.001,
        'train_samples': 1000,
        'val_samples': 200,
        'test_samples': 200,
    }
    
    print("="*60)
    print("集合划分问题 - 图对比学习模型")
    print("="*60)
    print(f"产品数: {config['m']}, 工序数: {config['n']}")
    print(f"节点数: {config['n_nodes']}, 集合数: {config['n_clusters']}")
    print(f"隐藏维度: {config['hidden_dim']}, GCN层数: {config['n_gcn_layers']}")
    print(f"使用对比学习: {config['use_contrastive']}")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = ClusteringDataset(
        config['train_samples'], config['m'], config['n'], 
        config['n_nodes'], config['n_clusters']
    )
    val_dataset = ClusteringDataset(
        config['val_samples'], config['m'], config['n'], 
        config['n_nodes'], config['n_clusters']
    )
    test_dataset = ClusteringDataset(
        config['test_samples'], config['m'], config['n'], 
        config['n_nodes'], config['n_clusters']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    # 创建模型
    print("\n创建模型...")
    model = ContrastiveClusteringModel(
        m=config['m'],
        n=config['n'],
        hidden_dim=config['hidden_dim'],
        n_gcn_layers=config['n_gcn_layers'],
        temperature=config['temperature'],
        use_contrastive=config['use_contrastive']
    ).to(device)
    
    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['n_epochs']
    )
    
    # 训练历史
    history = {
        'train_total_loss': [],
        'train_edge_loss': [],
        'train_contrast_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
    }
    
    best_val_f1 = 0.0
    
    # 训练循环
    print("\n开始训练...")
    print("="*60)
    
    for epoch in range(config['n_epochs']):
        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        
        # 记录训练损失
        history['train_total_loss'].append(train_losses['total'])
        history['train_edge_loss'].append(train_losses['edge'])
        if train_losses['contrast'] > 0:
            history['train_contrast_loss'].append(train_losses['contrast'])
        
        # 验证
        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(model, val_loader, device)
            
            # 记录验证指标
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            
            print(f"\nEpoch {epoch+1}/{config['n_epochs']}")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(Edge: {train_losses['edge']:.4f}, "
                  f"Contrast: {train_losses['contrast']:.4f})")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"  ✓ 保存最佳模型 (F1: {best_val_f1:.4f})")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    # 加载最佳模型进行测试
    print("\n在测试集上评估...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = evaluate(model, test_loader, device)
    
    print("\n测试集结果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_curves(history)
    
    print("\n全部完成!")
    
    return model, history, test_metrics


if __name__ == "__main__":
    model, history, test_metrics = main()
