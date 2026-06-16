import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerExtractor(BaseFeaturesExtractor):
    """
    基于Transformer的特征提取器，处理Seru和Batch的混合特征矩阵
    1. 使用可学习的实体嵌入
    2. 无位置编码设计
    3. 全局平均池化聚合特征
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # 解析观察空间维度：假设环境返回形状为 (12,3) 的观察矩阵
        self.num_entities = observation_space.shape[0]  # 12个实体（Seru + Batch）
        self.features_per_entity = observation_space.shape[1]  # 每个实体3个特征

        # 1. 类型嵌入层（对应环境中的特征0：1.0=Seru，0.0=Batch）
        self.type_embedding = nn.Embedding(
            num_embeddings=2,  # 两种实体类型（0和1）
            embedding_dim=32  # 将类型映射到32维向量
        )

        # 2. 时间特征编码器（处理特征1和2）
        self.time_feature_encoder = nn.Sequential(
            nn.Linear(self.features_per_entity - 1, 64),  # 输入2个时间特征（3-1=2）
            nn.ReLU(),  # 引入非线性
            nn.LayerNorm(64)  # 稳定训练过程
        )

        # 3. 精简Transformer配置（输入维度=32+64=96）
        self.transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=96,  # 输入维度（类型32+时间64）
                nhead=4,  # 4头注意力（原8头）
                dim_feedforward=256,  # FFN隐藏层维度（原512）
                batch_first=True  # 输入形状为(batch, seq, feature)
            ),
            num_layers=2  # 2层Transformer（原4层）
        )

        # 4. 最终投影层（将Transformer输出映射到目标维度）
        self.projection = nn.Linear(96, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: (batch_size, num_entities, features_per_entity)
        # 分离类型和时间特征
        # 类型在第0列；时间特征在第1:列
        type_indices = observations[:, :, 0].long()  # (batch_size, num_entities)
        time_features = observations[:, :, 1:]  # (batch_size, num_entities, features_per_entity-1)

        # 构造 mask：如果一个实体是 Batch（类型为0），且其所有时间特征均为 0，则认为该实体被屏蔽
        # mask: (batch_size, num_entities)，True 表示需要屏蔽
        mask = (type_indices == 0) & (time_features.abs().sum(dim=-1) == 0)

        # 类型嵌入：将类型转换成32维向量
        type_emb = self.type_embedding(type_indices)  # (batch_size, num_entities, 32)
        # 时间特征编码
        time_emb = self.time_feature_encoder(time_features)  # (batch_size, num_entities, 64)

        # 融合特征
        fused = th.cat([type_emb, time_emb], dim=-1)  # (batch_size, num_entities, 96)

        # Transformer：使用 src_key_padding_mask 参数屏蔽被mask的实体
        # 注意：src_key_padding_mask 的 shape 为 (batch_size, num_entities) 且为 bool，其中 True 表示需要忽略的元素
        encoded = self.transformer(fused, src_key_padding_mask=mask)  # (batch_size, num_entities, 96)

        # 使用 mask 进行全局平均池化，只对未屏蔽的实体取平均
        mask_float = (~mask).unsqueeze(-1).float()  # (batch_size, num_entities, 1)，未屏蔽为1，屏蔽为0
        encoded_masked = encoded * mask_float
        # 计算每个样本未屏蔽的实体数量，避免除以0
        valid_counts = mask_float.sum(dim=1).clamp(min=1)
        pooled = encoded_masked.sum(dim=1) / valid_counts  # (batch_size, 96)

        # 最终投影到目标维度
        return self.projection(pooled)
