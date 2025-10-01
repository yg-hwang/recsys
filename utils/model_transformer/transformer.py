import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union, Literal

from .layers import PositionalEncoding


class SimpleTransformer(nn.Module):
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

        super(SimpleTransformer, self).__init__()
        # multi-task 학습을 위한 task 수 (output_dims에 정의된 label 수)
        self.n_tasks = len(output_dims)

        # 시퀀스 길이와 pooling 방식 저장
        self.seq_len = seq_len
        self.global_pool = global_pool

        # -----------------------------------------------
        # Feature Embedding Layer
        # -----------------------------------------------
        # feature마다 별도의 nn.Embedding을 생성
        # (범주형 feature를 embedding_dim 차원 dense vector로 변환)
        self.embeddings = nn.ModuleDict(
            {
                feature_name: nn.Embedding(n_classes, embedding_dim)
                for feature_name, n_classes in feature_dims.items()
            }
        )

        # -----------------------------------------------
        # Transformer Encoder Layer
        # -----------------------------------------------
        # Transformer 기본 단위: self-attention + feedforward 블록
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads)

        # 위치 정보를 넣기 위한 Positional Encoding
        self.position_encoding = PositionalEncoding(
            dim_model=embedding_dim, max_len=seq_len
        )

        # Transformer encoder 레이어
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

        # -----------------------------------------------
        # Task-specific Output Tower
        # -----------------------------------------------
        # 각 타겟 변수별로 Linear layer 생성 (예: `y_color`의 vocab 크기만큼 출력 차원)
        # 각 예측 target label마다 출력 차원을 맞추기 위한 Linear layer 정의
        self.towers = nn.ModuleDict(
            {
                feature_name: nn.Linear(embedding_dim, n_classes)
                for feature_name, n_classes in output_dims.items()
            }
        )

    def _apply_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer 출력 시퀀스를 하나의 벡터로 요약
        입력: (seq_len, batch_size, hidden_dim)
        출력: (batch_size, hidden_dim)
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
        4) TransformerEncoder 적용 (self-attention)
        5) pooling -> user representation
        6) task tower를 통해 각 task 예측

        :param feature_sequences: {feature_name: 시퀀스 텐서}
        :param masks: padding mask (batch_size, seq_len)
        :return: (sequence vector, {target_name: 예측 로짓})
        """

        # -----------------------------------------------
        # Feature Embedding
        # -----------------------------------------------
        # feature별 임베딩 후 모두 합산
        # (batch_size, seq_len, embedding_dim)
        x_embed = sum(
            self.embeddings[feature_name](x)
            for feature_name, x in feature_sequences.items()
        )

        # -----------------------------------------------
        # Transformer 입력 형식 맞추기
        # -----------------------------------------------
        # Transformer는 (seq_len, batch_size, embedding_dim) 입력을 기대
        x_embed = x_embed.permute(1, 0, 2)

        # -----------------------------------------------
        # Positional Encoding 추가
        # -----------------------------------------------
        x_embed = self.position_encoding(x_embed)

        # -----------------------------------------------
        # Transformer Encoder 적용
        # -----------------------------------------------
        # src_key_padding_mask: 패딩 위치 무시 (batch_size, seq_len)
        x_embed = self.transformer(x_embed, src_key_padding_mask=masks)

        # -----------------------------------------------
        # Task별 예측 출력
        # -----------------------------------------------
        y_outputs = {}
        for target_name, tower in self.towers.items():
            # (seq_len, batch_size, embedding_dim) -> Linear -> (seq_len, batch_size, n_classes)
            y_outputs[target_name] = tower(x_embed)

        # -----------------------------------------------
        # Vector Representation (Pooling)
        # -----------------------------------------------
        # 최종 feature vector (batch_size, embedding_dim)
        x_final = self._apply_pooling(x_embed)

        return x_final, y_outputs
