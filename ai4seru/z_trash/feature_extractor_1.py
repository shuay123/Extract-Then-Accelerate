import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MinimalTransformerExtractor(BaseFeaturesExtractor):
    """
    极简Transformer特征提取器
    设计特点:
    1. 无升维词嵌入层
    2. 原始特征直入注意力层
    3. 后续全连接补偿
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # 输入参数验证
        self.num_entities = observation_space.shape[0]  # 201
        self.feature_dim = observation_space.shape[1]  # 3
        print(f"输入参数: 实体数={self.num_entities}, 特征维度={self.feature_dim}")

        # 1. 移除词嵌入层，直接使用原始特征
        self.raw_projection = nn.Linear(self.feature_dim, self.feature_dim)  # 保持维度不变

        # 2. 适配Transformer的配置
        self.transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=self.feature_dim,  # 保持原始特征维度(3)
                nhead=3,  # 头数需能被3整除
                dim_feedforward=64,  # 适度扩展
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )

        # 3. 增强型全连接补偿层
        self.post_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        处理流程:
        (B,201,3) → 投影 → (B,201,3) → Transformer → (B,201,3) → 池化 → (B,3) → FC → (B,128)
        """
        # 原始特征标准化
        x = self.raw_projection(observations)  # (B,201,3)

        # Transformer处理
        context = self.transformer(x)  # (B,201,3)

        # 动态重要性池化
        weights = th.sigmoid(context.mean(dim=-1, keepdim=True))  # (B,201,1)
        pooled = (context * weights).sum(dim=1)  # (B,3)

        # 全连接补偿
        return self.post_fc(pooled)  # (B,128)
