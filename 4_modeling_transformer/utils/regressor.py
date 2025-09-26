from torch import nn
from torch.nn import functional as F


class MultiOutputRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super(MultiOutputRegressor, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.config = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)

        x = self.output_layer(x)

        return x
