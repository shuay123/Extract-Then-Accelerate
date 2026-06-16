import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class EntitySelector(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.num_entities = observation_space.shape[0]  # 201
        self.feature_dim = observation_space.shape[1]  # 3

        # 1. 实体特征嵌入
        self.entity_embedding = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )

        # 2. Transformer编码器
        self.transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ),
            num_layers=2
        )

        # 3. 逐实体评分头
        self.scoring_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)  # 每个实体输出一个分数
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        输入: (B, 201, 3)
        输出: (B, 201) → 每个实体的选择概率
        """
        # 特征嵌入
        embeddings = self.entity_embedding(observations)  # (B,201,32)

        # Transformer编码
        context = self.transformer(embeddings)  # (B,201,32)

        # 逐实体评分
        scores = self.scoring_head(context).squeeze(-1)  # (B,201)

        return scores  # 未归一化的logits
