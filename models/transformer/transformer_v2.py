import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Literal, Union

from .layers import EmbeddingLayer, AttentionPooling, LearnablePositionalEncoding, MLP


class MultiTaskMoESequenceTransformer(nn.Module):
    """
    MoSE 기반 Multi-task Classification (MoSE 논문 응용 버전)
    - 여러 feature sequence를 입력으로 받아, 각 feature별 "다음 시퀀스"를 예측
    - pooling 결과 sequence_vector을 user vector로 사용

    응용 포인트
    ----------
    1) tower 제거
       - task별 추가 tower를 제거하고, `fused_seq` -> output head로 바로 logits 생성

    2) expert를 MLP로 변경
       - Transformer expert 대신 position-wise MLP expert 적용
       - 시간축(seq_len) mixing은 feature encoder 단계(Transformer)에서만 발생하고,
         expert는 각 timestep 벡터를 변환하는 역할만 수행 (연산량 대폭 감소)

    3) action weight 반영
        - `action_sequence`(클릭/찜/구매 등)와 `action_weights`가 주어지면,
          timestep별 가중치를 모든 feature token embedding에 곱해 행동 중요도를 반영

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
        global_pool: Literal["last", "avg", "max", "sum", "att"] = "last",
        # ---------- MoSE 관련 하이퍼파라미터 ---------- #
        n_experts: int = 4,
        expert_layers: int = 1,
        dropout: float = 0.0,
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

        :param n_experts:
            MoSE 구조에서 사용할 expert 개수

        :param expert_layers:
            MLP expert의 hidden layer 개수로 사용
            - 예: expert_layers=2 이면 [2 * embedding_dim, 2 * embedding_dim] 형태의 hidden 구성

        :param dropout:
            Transformer 내부 dropout 비율
        """

        super().__init__()

        if output_dims is None:
            raise ValueError("output_dims must be provided.")

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.global_pool = global_pool
        self.n_experts = n_experts

        # -----------------------------------------------
        # 입력 feature 구성
        # -----------------------------------------------
        # feature_dims에 "action"이 있어도 token embedding feature에서는 제외
        # action은 별도 `action_sequence`로 처리
        self.features = [k for k in feature_dims.keys() if k != "action"]

        # task(target) 이름은 `output_dims`의 key를 그대로 사용
        self.targets = list(output_dims.keys())

        # -----------------------------------------------
        # Feature Embedding Layer
        # -----------------------------------------------
        # feature마다 별도의 Embedding 생성
        # (범주형 feature를 embedding_dim 차원 dense vector로 변환)
        # 입력: (batch_size, seq_len)
        # 출력: (batch_size, seq_len, embedding_dim)
        self.embeddings = nn.ModuleDict(
            {
                name: EmbeddingLayer(int(n_classes), embedding_dim)
                for name, n_classes in feature_dims.items()
            }
        )

        # -----------------------------------------------
        # Action Weight Lookup (고정 가중치)
        # -----------------------------------------------
        self.action_weights = action_weights or {}
        if self.action_weights:
            max_action_id = max(int(k) for k in self.action_weights.keys())
            lookup = torch.ones(max_action_id + 1, dtype=torch.float32)
            for k, v in self.action_weights.items():
                lookup[int(k)] = float(v)
            # register_buffer: to(device) 시 같이 이동, optimizer 업데이트는 하지 않음
            self.register_buffer("action_lookup", lookup)
        else:
            self.action_lookup = None

        # -----------------------------------------------
        # Positional Encoding
        # -----------------------------------------------
        self.position_encoding = LearnablePositionalEncoding(
            dim_model=embedding_dim, max_len=seq_len
        )

        # # -----------------------------------------------
        # # Feature-wise Transformer Encoders
        # # -----------------------------------------------
        # base_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embedding_dim,
        #     nhead=n_heads,
        #     batch_first=True,
        #     dropout=dropout,
        # )
        #
        # self.feature_encoders = nn.ModuleDict(
        #     {
        #         feature_name: nn.TransformerEncoder(
        #             encoder_layer=base_encoder_layer,
        #             num_layers=n_layers,
        #         )
        #         for feature_name in self.features
        #     }
        # )

        # -----------------------------------------------
        # Shared Transformer Encoder
        # -----------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.shared_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

        # -----------------------------------------------
        # Shared Projection (feature concat -> shared_seq)
        # -----------------------------------------------
        # 입력: (batch_size, seq_len, embedding_dim * num_features)
        # 출력: (batch_size, seq_len, embedding_dim)
        # self.shared_proj = nn.Linear(embedding_dim * len(self.features), embedding_dim)

        # -----------------------------------------------
        # Experts (MoE)
        # -----------------------------------------------
        # Transformer 대신 position-wise MLP 로 사용 (연산량 감소를 위해)
        # 입력: (batch_size * seq_len, embedding_dim)
        # 출력: (batch_size * seq_len, embedding_dim)
        # hidden dims 구성:
        # - expert_layers=1 -> [2 * embedding_dim]
        # - expert_layers=2 -> [2 * embedding_dim, 2 * embedding_dim]
        hidden_dim = 2 * embedding_dim
        expert_hidden_dims = [hidden_dim for _ in range(max(1, int(expert_layers)))]
        self.experts = nn.ModuleList(
            [
                MLP(
                    input_dim=embedding_dim,
                    embedding_dims=expert_hidden_dims,
                    output_dim=embedding_dim,
                    dropout=dropout if dropout > 0 else None,
                    output_layer=True,
                )
                for _ in range(n_experts)
            ]
        )

        # -----------------------------------------------
        # Task-wise Gates
        # -----------------------------------------------
        # gate 입력은 `sequence_vector`(Pooling된 벡터)만 사용
        # 입력: (batch_size, embedding_dim)
        # 출력: (batch_size, n_experts)
        self.gates = nn.ModuleDict(
            {
                target_name: nn.Linear(embedding_dim, n_experts)
                for target_name in self.targets
            }
        )

        # -----------------------------------------------
        # Output Heads
        # -----------------------------------------------
        # 입력: (batch_size, seq_len, embedding_dim)
        # 출력: (batch_size, seq_len, n_classes)
        self.output_heads = nn.ModuleDict(
            {
                target_name: nn.Linear(embedding_dim, int(n_classes))
                for target_name, n_classes in output_dims.items()
            }
        )

        # -----------------------------------------------
        # Attention Pooling (옵션)
        # -----------------------------------------------
        if self.global_pool == "att":
            self.attention_pooling = AttentionPooling(hidden_dim=embedding_dim)

        # -----------------------------------------------
        # Causal Attention Mask (look-ahead 차단)
        # -----------------------------------------------
        # (seq_len, seq_len)
        # True  = attention 차단 (미래를 못 봄)
        # False = attention 허용
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_attn_mask", causal_mask)

    def _compute_action_weight_tensor(
        self, action_sequence: torch.Tensor, masks: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Action weight tensor 생성

        :param action_sequence: (batch_size, seq_len)
        :param masks: (batch_size, seq_len)
        :param dtype: embedding dtype에 맞추기 위함
        :return: weights: (batch_size, seq_len, 1)
        """

        if masks.dtype != torch.bool:
            masks = masks.to(torch.bool)

        # lookup이 없으면 전부 1.0
        if self.action_lookup is None:
            return torch.ones(
                (action_sequence.size(0), action_sequence.size(1), 1),
                device=action_sequence.device,
                dtype=dtype,
            )

        # action id가 lookup 범위를 넘으면 안전하게 확장 (기본값 1.0)
        max_id_in_batch = int(action_sequence.max().item())
        if max_id_in_batch >= self.action_lookup.size(0):
            new_lookup = torch.ones(
                max_id_in_batch + 1,
                device=self.action_lookup.device,
                dtype=self.action_lookup.dtype,
            )
            new_lookup[: self.action_lookup.size(0)] = self.action_lookup
            self.action_lookup = new_lookup

        # weights: (batch_size, seq_len, 1)
        weights = (
            self.action_lookup[action_sequence.long()].to(dtype=dtype).unsqueeze(-1)
        )

        # padding 위치는 embedding 곱에서 왜곡 방지용으로 1.0 고정
        weights = weights.masked_fill(masks.unsqueeze(-1), 1.0)

        return weights

    def _apply_pooling(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Transformer 출력 시퀀스를 하나의 벡터로 요약
        :param x: (batch_size, seq_len, embedding_dim)
        :param masks: (batch_size, seq_len)
        :return: (batch_size, embedding_dim)
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
        feature_sequences: Dict[str, torch.Tensor],
        action_sequence: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param feature_sequences: {feature_name: 시퀀스 텐서}
        :param action_sequence: (선택) 행동 시퀀스 텐서
        :param masks: padding mask (batch_size, seq_len)
        :return:
          - sequence_vector: (batch_size, embedding_dim)
          - y_outputs: {target_name: (batch_size, seq_len, n_classes)}
        """

        if masks is None:
            raise ValueError(
                "masks must be provided. (batch_size, seq_len) True=padding"
            )

        if masks.dtype != torch.bool:
            masks = masks.to(torch.bool)

        batch_size = masks.size(0)
        seq_len = masks.size(1)
        embedding_dim = self.embedding_dim

        # -------------------------------------------
        # Causal Attention Mask
        # -------------------------------------------
        # (seq_len, seq_len)
        causal_attn_mask = self.causal_attn_mask.to(device=masks.device)

        # -------------------------------------------
        # action weight (선택)
        # -------------------------------------------
        # weights: (batch_size, seq_len, 1)
        if action_sequence is not None and self.action_lookup is not None:
            weights = self._compute_action_weight_tensor(
                action_sequence=action_sequence,
                masks=masks,
                dtype=torch.float32,
            )
        else:
            weights = None

        # # -------------------------------------------
        # # feature별 encoding (미래 토큰 차단 적용)
        # # -------------------------------------------
        # encoded_list = []
        #
        # for feature_name in self.features:
        #     # token embedding
        #     # 입력: (batch_size, seq_len)
        #     # 출력: (batch_size, seq_len, embedding_dim)
        #     x_embed = self.embeddings[feature_name](feature_sequences[feature_name])
        #
        #     # action weight 적용 (고정 가중치)
        #     if weights is not None:
        #         x_embed = x_embed * weights.to(dtype=x_embed.dtype)
        #
        #     # positional encoding
        #     x_embed = self.position_encoding(x_embed)
        #
        #     # Transformer Encoder 적용
        #     # - src_key_padding_mask: padding 무시 (batch_size, seq_len)
        #     # - mask(causal_attn_mask): 미래 토큰 차단 (seq_len, seq_len)
        #     x_embed = self.feature_encoders[feature_name](
        #         x_embed,
        #         mask=causal_attn_mask,
        #         src_key_padding_mask=masks,
        #     )
        #
        #     encoded_list.append(x_embed)
        #
        # # -------------------------------------------
        # # shared sequence 생성
        # # -------------------------------------------
        # # concat: (batch_size, seq_len, embedding_dim * num_features)
        # # proj:   (batch_size, seq_len, embedding_dim)
        # shared_seq = torch.cat(encoded_list, dim=-1)
        # shared_seq = self.shared_proj(shared_seq)

        # -------------------------------------------
        # Shared embedding (feature embedding 합산)
        # -------------------------------------------
        # x_embed: (batch_size, seq_len, embedding_dim)
        x_embed = 0.0
        for feature_name in self.features:
            # token embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            f_embed = self.embeddings[feature_name](feature_sequences[feature_name])

            # action weight 적용 (선택): (batch_size, seq_len, embedding_dim) * (batch_size, seq_len, 1)
            if weights is not None:
                f_embed = f_embed * weights.to(dtype=f_embed.dtype)

            x_embed = x_embed + f_embed

        # positional encoding: (batch_size, seq_len, embedding_dim)
        x_embed = self.position_encoding(x_embed)

        # -------------------------------------------
        # Shared Transformer Encoder 적용 (미래 토큰 차단)
        # -------------------------------------------
        # shared_seq: (batch_size, seq_len, embedding_dim)
        shared_seq = self.shared_transformer_encoder(
            x_embed,
            mask=causal_attn_mask,  # (seq_len, seq_len)
            src_key_padding_mask=masks,  # (batch_size, seq_len)
        )

        # -------------------------------------------
        # pooling -> sequence representation
        # -------------------------------------------
        # sequence_vector: (batch_size, embedding_dim)
        sequence_vector = self._apply_pooling(shared_seq, masks)

        # -------------------------------------------
        # experts (MLP expert, position-wise)
        # -------------------------------------------
        # shared_seq: (batch_size, seq_len, embedding_dim)
        # flat:       (batch_size * seq_len, embedding_dim)
        flat = shared_seq.reshape(batch_size * seq_len, embedding_dim)

        # expert_outputs[i]: (batch_size, seq_len, embedding_dim)
        expert_outputs = []
        for expert in self.experts:
            out = expert(flat)  # (batch_size * seq_len, embedding_dim)
            out = out.reshape(batch_size, seq_len, embedding_dim)
            expert_outputs.append(out)

        # expert_stack: (batch_size, seq_len, n_experts, embedding_dim)
        expert_stack = torch.stack(expert_outputs, dim=2)

        # -------------------------------------------
        # task-wise routing + prediction (tower 제거)
        # -------------------------------------------
        y_outputs: Dict[str, torch.Tensor] = {}

        for target_name in self.targets:
            # gate logits
            # 입력: (batch_size, embedding_dim)
            # 출력: (batch_size, n_experts)
            gate_logits = self.gates[target_name](sequence_vector)
            gate_weights = torch.softmax(gate_logits, dim=-1)

            # ---------------------------------------
            # fused sequence = Σ_k gate_k * expert_k
            # ---------------------------------------
            # expert_stack: (batch_size, seq_len, n_experts, embedding_dim)
            # gate_weights: (batch_size, n_experts)
            #
            # gate_weights를 broadcasting 가능한 shape으로 확장
            # (batch_size, n_experts) -> (batch_size, 1, n_experts, 1)
            gate_expand = gate_weights.unsqueeze(1).unsqueeze(-1)

            # (batch_size, seq_len, n_experts, embedding_dim) -> sum over experts
            # fused_seq: (batch_size, seq_len, embedding_dim)
            fused_seq = (expert_stack * gate_expand).sum(dim=2)

            # logits
            # 입력: (batch_size, seq_len, embedding_dim)
            # 출력: (batch_size, seq_len, n_classes)
            y_outputs[target_name] = self.output_heads[target_name](fused_seq)

        # -------------------------------------------
        # sequence vector (as user vector)
        # -------------------------------------------
        # sequence_vector: (batch_size, embedding_dim)
        return sequence_vector, y_outputs
