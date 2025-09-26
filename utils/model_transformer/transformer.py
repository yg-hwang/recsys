import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union, Literal

from .layers import PositionalEncoding


class SimpleTransformerRec(nn.Module):
    """
    TransformerEncoder 기반 Sequential Recommendation (Baseline)
    - 여러 feature sequence를 임베딩하여 Transformer로 학습
    - 마지막에 pooling 후, task별 예측 head(tower)를 통해 결과 출력
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        embedding_dim: int = 64,
        seq_len: int = 10,
        n_heads: int = 2,
        n_layers: int = 2,
        output_dims: Dict[str, int] = None,
        global_pool: Literal["last", "avg", "max", "sum"] = "sum",
    ):
        """
        :param feature_dims: 입력 feature 이름과 feature별 클래스 개수
        :param embedding_dim: 각 feature를 임베딩할 차원 크기(Transformer Encoder 임베딩 차원과 동일)
        :param seq_len: 시퀀스 길이
        :param n_heads: Transformer multi-head attention 개수
        :param n_layers: Transformer Encoder 레이어 수
        :param output_dims: 츨략 Label 클래스 개수
        :param global_pool: 임베딩 값 출력 Pooling 방식
        """

        super(SimpleTransformerRec, self).__init__()
        # multi-task 학습을 위한 task 수 (output_dims에 정의된 label 수)
        self.n_tasks = len(output_dims)

        # 시퀀스 길이와 pooling 방식 저장
        self.seq_len = seq_len
        self.global_pool = global_pool

        # 각 feature 마다 임베딩 레이어 생성 (예: item_id, category_id)
        # nn.Embedding: 범주형 값을 dense vector로 변환
        self.embeddings = nn.ModuleDict(
            {
                feature_name: nn.Embedding(n_classes, embedding_dim)
                for feature_name, n_classes in feature_dims.items()
            }
        )

        # TransformerEncoder 기본 단위: self-attention + feedforward 블록
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads)

        # 위치 정보가 없는 Transformer에 시퀀스 내 순서를 입력
        self.position_encoding = PositionalEncoding(
            dim_model=embedding_dim, max_len=seq_len
        )

        # Transformer encoder 레이어
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

        # 각 예측 label(target 변수)마다 출력 차원을 맞추기 위한 Linear layer 정의
        self.towers = nn.ModuleDict(
            {
                feature_name: nn.Linear(embedding_dim, n_classes)
                for feature_name, n_classes in output_dims.items()
            }
        )

    def _apply_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer 출력 시퀀스를 하나의 벡터로 요약하는 함수
        (seq_len, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        """

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
        Forward 계산 과정:
        1) feature별 임베딩
        2) feature 임베딩 합산 (같은 길이 시퀀스로 통합)
        3) Positional Encoding 추가
        4) TransformerEncoder 적용
        5) pooling -> user representation
        6) task tower를 통해 각 task 예측

        :param feature_sequences: list of input sequences
        :param masks: list of padding masks for each input sequence
        """

        # (batch_size, seq_len, embedding_dim) feature별 임베딩 합산
        x_embed = sum(
            self.embeddings[feature_name](x)
            for feature_name, x in feature_sequences.items()
        )

        # Transformer 입력은 (seq_len, batch_size, embedding_dim) 형태여야 함
        x_embed = x_embed.permute(1, 0, 2)

        # 순서 정보를 넣기 위해 positional encoding 더해줌
        x_embed = self.position_encoding(x_embed)

        # Transformer encoder 통과 (self-attention 수행)
        # src_key_padding_mask: (batch_size, seq_len) -> padding mask 적용 가능
        x_embed = self.transformer(x_embed, src_key_padding_mask=masks)

        # 각 task별 출력 계산
        y_outputs = {}
        for target_name, tower in self.towers.items():
            # (seq_len, batch_size, embedding_dim) -> Linear -> (seq_len, batch_size, n_classes)
            # y_output: (seq_len, batch_size, n_classes)
            y_outputs[target_name] = tower(x_embed)

        # 최종 feature vector 출력 (pooling 적용)
        # (seq_len, batch_size, embedding_dim) -> (batch_size, embedding_dim)
        x_final = self._apply_pooling(x_embed)

        return x_final, y_outputs
