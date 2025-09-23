import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union, Literal

from .layers import PositionalEncoding


class SimpleTransformerRec(nn.Module):
    """
    TransformerEncoder for Sequential Recommendation (Baseline)
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        embedding_dim: int = 64,
        seq_len: int = 10,
        n_heads: int = 2,
        n_layers: int = 2,
        y_output_dims: Dict[str, int] = None,
        global_pool: Literal["last", "avg", "max", "sum"] = "sum",
    ):
        """
        :param feature_dims: 각 시퀀스 feature 입력 차원
        :param embedding_dim: Transformer Encoder 임베딩 차원
        :param seq_len: 시퀀스 길이
        :param n_heads: Attention head
        :param n_layers: Transformer Encoder 레이어 수
        :param y_output_dims: Target Label 클래스 수
        :param global_pool: 임베딩 값 출력 방식
        """

        super(SimpleTransformerRec, self).__init__()
        self.n_tasks = len(y_output_dims)
        self.seq_len = seq_len
        self.global_pool = global_pool
        self.embeddings = nn.ModuleDict(
            {
                feature_name: nn.Embedding(n_classes, embedding_dim)
                for feature_name, n_classes in feature_dims.items()
            }
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads)
        self.position_encoding = PositionalEncoding(
            dim_model=embedding_dim, max_len=seq_len
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )
        self.towers = nn.ModuleDict(
            {
                feature_name: nn.Linear(embedding_dim, n_classes)
                for feature_name, n_classes in y_output_dims.items()
            }
        )

    def _apply_pooling(self, x: torch.Tensor) -> torch.Tensor:
        match self.global_pool:
            case "last":
                return x[-1]
            case "avg":
                return torch.mean(x, dim=0)
            case "max":
                x, _ = torch.max(x, dim=0)
                return x
            case "sum":
                return torch.sum(x, dim=0)
            case _:
                raise ValueError(
                    "`global_pool` must be 'last', 'avg', 'max', or 'sum'."
                )

    def forward(
        self,
        feature_sequences: Dict[str, torch.Tensor],
        masks: Union[List[torch.Tensor], torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param feature_sequences: list of input sequences
        :param masks: list of padding masks for each input sequence
        """

        # x_embed: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        x_embed = sum(
            self.embeddings[feature_name](x)
            for feature_name, x in feature_sequences.items()
        )

        # (batch_size, seq_len, embedding_dim) -> (seq_len, batch_size, embedding_dim)
        x_embed = x_embed.permute(1, 0, 2)

        # (seq_len, batch_size, embedding_dim) -> (seq_len, batch_size, embedding_dim)
        x_embed = self.position_encoding(x_embed)

        # Apply `src_key_padding_mask` (batch_size, seq_len)
        x_embed = self.transformer(x_embed, src_key_padding_mask=masks)

        y_outputs = {}
        for target_name, tower in self.towers.items():
            # y_output: (seq_len, batch_size, n_classes)
            y_outputs[target_name] = tower(x_embed)
        # x_final: (batch_size, embedding_dim)
        x_final = self._apply_pooling(x_embed)

        return x_final, y_outputs
