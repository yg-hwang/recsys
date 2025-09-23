import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_len: int = 20):
        super(PositionalEncoding, self).__init__()

        # pos = [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        # div_term = 10000^(2i/dim_model)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / dim_model)
        )  # (dim_model // 2)

        # Compute sin(pos) and cos(pos) for even/odd indices
        pe = torch.zeros(max_len, dim_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices (2i)
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices (2i+1)

        # Add a batch dimension for broadcasting
        pe = pe.unsqueeze(1)  # (max_len, 1, dim_model)
        self.register_buffer("pe", pe)  # Save as a buffer, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (seq_len, batch_size, embedding_dim)
        """
        # Position encoding은 입력 x에 더해줌 (broadcast)
        x = x + self.pe[: x.size(0), :]

        return x
