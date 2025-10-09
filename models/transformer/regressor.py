from torch import nn
from torch.nn import functional as F


class MultiOutputRegressor(nn.Module):
    """
    시퀀스 벡터를 아이템 벡터 공간으로 projection하는 MLP 기반 회귀 모델
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super(MultiOutputRegressor, self).__init__()

        # 다층 퍼셉트론 구조
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 정규화 & 일반화
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # config 저장 (재현성 및 추후 로깅용)
        self.config = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Forward: 시퀀스 벡터를 아이템 벡터로 출력 변환
        """

        # Linear -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)

        # Linear -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)

        # Linear -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)

        # 출력층: Linear (64차원 아이템 벡터로 매핑)
        x = self.output_layer(x)
        x = F.normalize(x, dim=-1)

        return x
