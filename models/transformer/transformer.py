import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union, Literal

from .layers import EmbeddingLayer, LearnablePositionalEncoding


class MultiTaskSequenceTransformer(nn.Module):
    """
    Shared TransformerEncoder 기반 Multi-task Classification (Baseline)
    - 여러 feature sequence를 임베딩하여 Transformer로 학습 (behavior sequence transformer과 유사)
    - 마지막에 pooling 후, task별 예측 head(tower)를 통해 결과 출력
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        action_weights: Dict[int, Union[int, float]] = None,
        embedding_dim: int = 64,
        seq_len: int = 10,
        n_heads: int = 2,
        n_layers: int = 2,
        output_dims: Dict[str, int] = None,
        global_pool: Literal["last", "avg", "max", "sum"] = "last",
    ):
        """
        :param feature_dims:
            입력 feature 이름과 feature별 클래스 개수
            예) {"color": 10, "price": 8, "category": 20, "action": 4}

        :param action_weights:
            행동 가중치
            예) {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
            - action_sequence가 주어질 경우 timestep별로 embedding에 곱해짐
            - padding 위치에는 영향을 주지 않음

        :param embedding_dim:
            각 feature를 임베딩할 차원 크기 (Transformer Encoder 임베딩 차원과 동일)

        :param seq_len:
            입력 시퀀스 길이 (모델 내부에서 causal attention mask 생성에 사용)

        :param n_heads:
            Transformer multi-head attention 개수

        :param n_layers:
            Feature별 Transformer Encoder 레이어 수

        :param output_dims:
            출력 target label과 클래스 개수
            예) {"color": 10, "price": 8, "category": 20}

        :param global_pool:
            시퀀스 임베딩을 하나의 feature vector 형태로 요약하는 pooling 방식
            - "last": 마지막 유효 토큰
            - "avg": 평균 pooling (padding 제외)
            - "sum": 합 pooling (padding 제외)
            - "max": max pooling (padding 제외)
            - "att": attention pooling
        """

        super(MultiTaskSequenceTransformer, self).__init__()
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
                name: EmbeddingLayer(n_classes, embedding_dim)
                for name, n_classes in feature_dims.items()
            }
        )

        # -----------------------------------------------
        # Positional Encoding
        # -----------------------------------------------
        self.position_encoding = LearnablePositionalEncoding(
            dim_model=embedding_dim, max_len=seq_len
        )

        # -----------------------------------------------
        # Shared Transformer Encoder
        # -----------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, batch_first=True
        )

        # Transformer encoder
        self.shared_transformer_encoder = nn.TransformerEncoder(
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

        # -----------------------------------------------
        # Action Weight Lookup (고정 가중치)
        # -----------------------------------------------
        self.action_weights = action_weights or {}

        # -----------------------------------------------
        # Causal Attention Mask 준비 (look-ahead 차단)
        # -----------------------------------------------
        # (seq_len, seq_len)
        # True  = attention 차단 (미래를 못 봄)
        # False = attention 허용
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_attn_mask", causal_mask)

    def _apply_pooling(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Transformer 출력 시퀀스를 하나의 벡터로 요약
        입력: (batch_size, seq_len, embedding_dim)
        출력: (batch_size, embedding_dim)
        """

        if self.global_pool == "last":
            # 각 배치의 마지막 유효 토큰 인덱스 계산 (batch,)
            valid_mask = (~masks).unsqueeze(-1).float()
            masked_x = x * valid_mask
            # 음수 방지
            valid_lengths = valid_mask.sum(dim=1).squeeze(-1)
            last_indices = (valid_lengths - 1).clamp(min=0).long()
            batch_indices = torch.arange(x.size(0), device=x.device)
            # (batch_size, embedding_dim)
            return masked_x[batch_indices, last_indices]

        elif self.global_pool == "avg":
            # 유효 토큰만 평균
            valid = (~masks).unsqueeze(-1)  # (batch_size, seq_len, 1)
            sum_x = (x * valid).sum(dim=1)  # (batch_size, embedding_dim)
            denom = valid.sum(dim=1).clamp(min=1)  # (batch_size, 1)
            return sum_x / denom

        elif self.global_pool == "sum":
            # 유효 토큰만 합산
            valid = (~masks).unsqueeze(-1)  # (batch_size, seq_len, 1)
            return (x * valid).sum(dim=1)  # (batch_size, embedding_dim)

        elif self.global_pool == "max":
            x_masked = x.masked_fill(masks.unsqueeze(-1), float("-inf"))
            return x_masked.max(dim=1).values  # (batch_size, embedding_dim)

        elif self.global_pool == "att":
            return self.attention_pooling(x, mask=masks)

        else:
            raise ValueError(
                "`global_pool` must be 'last', 'avg', 'max', 'sum', or 'att'."
            )

    def forward(
        self,
        feature_sequence: Dict[str, torch.Tensor],
        action_sequence: torch.Tensor = None,
        masks: Union[List[torch.Tensor], torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward 계산 과정:
        1) feature별 임베딩
        2) feature 임베딩 합산 (같은 길이 시퀀스로 통합)
        3) Positional Encoding 추가
        4) TransformerEncoder 적용 (self-attention)
           - src_key_padding_mask: padding 무시
           - causal mask: 미래 토큰 차단
        5) pooling -> user representation
        6) task tower를 통해 각 task 예측

        :param feature_sequence: {feature_name: 시퀀스 텐서}
        :param action_sequence: (선택) 행동 시퀀스 텐서
        :param masks: padding mask (batch_size, seq_len)
        :return:
          - sequence_vector: (batch_size, embedding_dim)
          - y_outputs: {target_name: (batch_size, seq_len, n_classes)}
        """

        if masks.dtype != torch.bool:
            masks = masks.to(torch.bool)

        # -----------------------------------------------
        # Feature Embedding
        # -----------------------------------------------
        # feature별 임베딩 후 모두 합산
        # (batch_size, seq_len, embedding_dim)
        x_embed = sum(
            self.embeddings[feature_name](x)
            for feature_name, x in feature_sequence.items()
        )

        # -----------------------------------------------
        # Action Weight 적용 (고정 가중치)
        # -----------------------------------------------
        # action weight는 token embedding에 timestep별로 곱함
        # padding 위치는 왜곡 방지를 위해 weight=1.0으로 고정
        if action_sequence is not None and self.action_weights:
            # 행동 ID -> 가중치 값 매핑용 룩업 벡터 생성
            action_indices = [int(k) for k in self.action_weights.keys()]
            max_id = max(max(action_indices), int(action_sequence.max().item()))
            lookup = torch.ones(max_id + 1, device=x_embed.device, dtype=x_embed.dtype)
            for k, v in self.action_weights.items():
                lookup[int(k)] = float(v)

            # (batch, seq_len) -> (batch, seq_len, 1) broadcasting 곱
            weights = lookup[action_sequence.long()].unsqueeze(-1)

            # padding 위치는 weight=1.0 고정
            weights = weights.masked_fill(masks.unsqueeze(-1), 1.0)

            x_embed = x_embed * weights

        # -----------------------------------------------
        #  Transformer 입력 변환
        # -----------------------------------------------
        # batch_first=True이므로 permute 불필요
        # (batch_size, seq_len, embedding_dim)
        x_embed = self.position_encoding(x_embed)

        # Transformer Encoder 적용
        # src_key_padding_mask: 패딩 위치 무시 (batch_size, seq_len)
        # causal_attn_mask: 미래 토큰 차단 (seq_len, seq_len)
        causal_attn_mask = self.causal_attn_mask.to(device=x_embed.device)
        x_embed = self.shared_transformer_encoder(
            x_embed, mask=causal_attn_mask, src_key_padding_mask=masks
        )

        # -----------------------------------------------
        # Task별 예측 출력
        # -----------------------------------------------
        y_outputs = {}
        for target_name, tower in self.towers.items():
            # (batch_size, seq_len, embedding_dim) -> Linear -> (seq_len, batch_size, n_classes)
            y_outputs[target_name] = tower(x_embed)

        # -----------------------------------------------
        # Vector Representation (Pooling)
        # -----------------------------------------------
        # 최종 sequence vector (batch_size, embedding_dim)
        x_final = self._apply_pooling(x_embed, masks)

        return x_final, y_outputs
