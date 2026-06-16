import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BatchNormNode(nn.Module):
    """节点批归一化"""
    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        # x: [B, N, D]
        x_trans = x.transpose(1, 2).contiguous()  # [B, D, N]
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # [B, N, D]
        return x_bn


class NodeFeatures(nn.Module):
    """节点特征更新模块"""
    def __init__(self, hidden_dim):
        super(NodeFeatures, self).__init__()
        self.node_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.to_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.edge_embedding = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e, edge_index, n_edges):
        # x: [B, N, D], e: [B, N*(N-1), D], edge_index: [B, N*(N-1)]
        batch_size, num_nodes, hidden_dim = x.size()
        
        Ux = self.node_embedding(x)  # [B, N, D]
        Vx = self.to_embedding(x)     # [B, N, D]
        Ve = self.edge_embedding(e)   # [B, N*(N-1), D]
        
        # 边注意力: [B, N, N-1, D]
        Ve = F.softmax(Ve.view(batch_size, num_nodes, n_edges, hidden_dim), dim=2)
        Ve = Ve.view(batch_size, num_nodes * n_edges, hidden_dim)  # [B, N*(N-1), D]
        
        # 获取目标节点特征: [B, N*(N-1), D]
        Vx = Vx[torch.arange(batch_size).view(-1, 1), edge_index]
        
        # 加权聚合
        to = Ve * Vx  # [B, N*(N-1), D]
        to = to.view(batch_size, num_nodes, n_edges, hidden_dim).sum(2)  # [B, N, D]
        
        x_new = Ux + to
        return x_new


class EdgeFeatures(nn.Module):
    """边特征更新模块"""
    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.hidden_dim = hidden_dim
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_from = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_to = nn.Linear(hidden_dim, hidden_dim, True)
        self.inverse_U = nn.Linear(hidden_dim, hidden_dim, True)
        self.W_placeholder = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_placeholder.data.uniform_(-1, 1)

    def forward(self, x, e, edge_index, inverse_edge_index, n_edges):
        # x: [B, N, D], e: [B, N*(N-1), D]
        batch_size, graph_size, hidden_dim = x.size()
        
        Ue = self.U(e)  # [B, N*(N-1), D]
        inverse_Ue = self.inverse_U(e)  # [B, N*(N-1), D]
        
        # 添加占位符
        inverse_Ue = torch.cat(
            (inverse_Ue, self.W_placeholder.view(1, 1, hidden_dim).repeat(batch_size, 1, 1)), 
            1
        )  # [B, N*(N-1)+1, D]
        
        inverse_node_embedding = inverse_Ue[
            torch.arange(batch_size).view(batch_size, 1), 
            inverse_edge_index
        ]  # [B, N*(N-1), D]
        
        Vx_from = self.V_from(x)  # [B, N, D]
        Vx_to = self.V_to(x)      # [B, N, D]
        Vx = Vx_to[torch.arange(batch_size).view(-1, 1), edge_index]  # [B, N*(N-1), D]
        
        Vx = Vx.view(batch_size, -1, n_edges, self.hidden_dim) + \
             Vx_from.view(batch_size, -1, 1, self.hidden_dim)
        Vx = Vx.view(batch_size, -1, self.hidden_dim)  # [B, N*(N-1), D]
        
        e_new = Ue + Vx + inverse_node_embedding
        return e_new


class SparseGCNLayer(nn.Module):
    """稀疏图卷积层"""
    def __init__(self, hidden_dim):
        super(SparseGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormNode(hidden_dim)

    def forward(self, x, e, edge_index, inverse_edge_index, n_edges):
        # x: [B, N, D], e: [B, N*(N-1), D]
        e_in = e
        x_in = x

        x_tmp = self.node_feat(x_in, e_in, edge_index.long(), n_edges)
        x_tmp = self.bn_node(x_tmp)
        x = F.relu(x_tmp)
        x_new = x_in + x  # [B, N, D]

        e_tmp = self.edge_feat(x_new, e_in, edge_index.long(), 
                               inverse_edge_index.long(), n_edges)
        e_tmp = self.bn_edge(e_tmp)
        e = F.relu(e_tmp)
        e_new = e_in + e  # [B, N*(N-1), D]
        
        return x_new, e_new