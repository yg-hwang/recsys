# =================================================
# `gate_input` 파라미터로 gating 방식 선택할 수 있는 코드
# =================================================

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal, Union
from torch import Tensor

from .layers import EmbeddingLayer, AttentionPooling, LearnablePositionalEncoding


class TransformerExpert(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        nhead: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # -------------------------------------------------
        # Expert 내부는 TransformerEncoder로 구성
        # - shared_seq(batch_size, seq_len, embedding_dim)를 입력으로 받아
        # - expert별로 서로 다른 변환을 수행한 출력(batch_size, seq_len, embedding_dim)을 만듦
        # -------------------------------------------------
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # 깊어져도 학습 안정성에 유리
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # -------------------------------------------------
        # residual + layer norm을 명시적으로 한 번 더 적용
        # - encoder 내부에도 residual이 있지만, 여기서는 expert 전체를 하나의 블록으로 보고
        # - "입력 x + expert_out" 형태로 안정적인 업데이트를 보장
        # -------------------------------------------------
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask=None, attn_mask=None
    ) -> torch.Tensor:
        # -------------------------------------------------
        # Expert TransformerEncoder
        # - mask=attn_mask: future token을 보지 못하도록 causal 차단
        # - src_key_padding_mask: padding token은 attention에서 무시
        # -------------------------------------------------

        # x: (batch_size, seq_len, embedding_dim)
        out = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

        # -------------------------------------------------
        # residual + layer norm
        # - Expert 변환(out)을 입력(x)에 더해 업데이트하고
        # - LayerNorm으로 스케일을 안정화
        # -------------------------------------------------
        return self.ln(x + out)


class MultiTaskMoESequenceTransformer(nn.Module):
    """
    MoSE 기반 Multi-task Classification (MoSE 논문 응용 버전)
    - 여러 feature sequence를 입력으로 받아, 각 feature별 "다음 시퀀스"를 예측
    - LSTM 대신 TransformerEncoder로 변경
    - non-sequential feature 반영
    - action weight 반영
        - `action_sequence`(클릭/찜/구매 등)와 `action_weights`가 주어지면,
           timestep별 가중치를 모든 feature token embedding에 곱해 행동 중요도를 반영
    - pooling된 `sequence_vector`을 user vector로 사용
    """

    def __init__(
        self,
        feature_sequence_dims: Dict[str, int],
        action_weights: Dict[int, Union[int, float]] = None,
        # ---------- TransformerEncoder 하이퍼파라미터 ---------- #
        embedding_dim: int = 64,
        seq_len: int = 10,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.0,
        output_dims: Dict[str, int] = None,
        global_pool: Literal["last", "avg", "max", "sum", "att"] = "last",
        # ---------- MoE 관련 하이퍼파라미터 ---------- #
        n_experts: int = 4,
        expert_layers: int = 1,
        expert_nhead: int = 1,
        top_k: int = None,
        gate_input: Literal["timestep", "sequence"] = "timestep",
        # ---------- non-sequential feature ---------- #
        feature_sparse_dims: Optional[Dict[str, int]] = None,
        feature_dense_dims: Optional[Dict[str, int]] = None,
        dense_hidden: int = 64,
    ):
        """
        :param feature_sequence_dims:
            입력 시퀀스 feature 이름과 클래스 개수
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
            Shared Transformer Encoder 레이어 수

        :param dropout:
            Transformer multi-head attention dropout 비율

        :param output_dims:
            출력 target label과 클래스 개수
            예) {"color": 10, "price": 8, "category": 20}

        :param global_pool:
            시퀀스 임베딩을 하나의 feature vector 형태로 요약하는 pooling 방식

        :param n_experts:
            MoE 구조에서 사용할 expert 개수

        :param expert_layers:
            각 TransformerExpert의 num_layers로 사용

        :param expert_nhead:
            각 TransformerExpert의 head 수 (가볍게 1~2 head로 두는 편이 비용이 적음)

        :param top_k:
            gating에서 상위 k개의 expert만 사용하도록 강제하는 sparsity 옵션
            - k가 작을수록 계산을 줄이고 분업을 강제

        :param gate_input:
            gate 입력 모드
            - "timestep": 각 timestep마다 expert를 gating
                - 시퀀스 길이가 길거나 다양할 때 선택하여 각 timestep 별로 Expert 라우팅을 수행
            - "sequence": 전체 sequence를 하나의 feature로 인식하여 expert를 gating
                - 시퀀스 길이가 짧을 땐 Expert 라우팅 안정성을 강화할 수 있음
        """

        super().__init__()

        if output_dims is None:
            raise ValueError("output_dims must be provided.")

        # 시퀀스 길이와 pooling 방식 저장
        self.seq_len = seq_len
        self.global_pool = global_pool
        self.embedding_dim = embedding_dim

        # Gate 입력 방식
        if gate_input not in ["timestep", "sequence"]:
            raise ValueError("gate_input must be 'timestep' or 'sequence'.")
        self.gate_input = gate_input

        # TransformerExpert의 head 수
        self.expert_nhead = expert_nhead

        # TransformerExpert 수
        self.n_experts = n_experts

        # -------------------------------------------------
        # Top-K Gating
        # - 원래 의도는 전체 expert 개수 보다 적은 개수만 gating에 사용하는 것.
        # - 그런데 expert가 8개 이하로 적을 때는 오히려 전체를 사용하는 게 학습 안정성이 높을 수 있으므로,
        # - 'top_k'가 None이면 전체 expert를 사용함.
        # -------------------------------------------------
        if top_k is not None:
            top_k = int(top_k)
            if top_k < 1:
                raise ValueError("top_k must be >= 1")
        self.top_k = n_experts if top_k is None else min(top_k, n_experts)

        # expert layer는 최소 1을 보장
        self.expert_num_layers = max(1, int(expert_layers))

        # -------------------------------------------------
        # gate temperature (학습 가능)
        # - softmax(logits / temperature)
        # - temperature ↓ : 특정 expert에 몰아주는 routing (샤프)
        # - temperature ↑ : 더 고르게 섞는 routing (소프트)
        # -------------------------------------------------
        self.gate_temperature = nn.Parameter(torch.tensor(1.0))
        self.gate_temp_min = 0.5  # 너무 샤프해지는 것 방지
        self.gate_temp_max = 2.0  # 너무 퍼지는 것 방지

        # gate 입력 안정화
        self.gate_ln = nn.LayerNorm(self.embedding_dim)

        # -----------------------------------------------
        # 입력 feature 구성
        # - feature_sequence_dims에 "action"이 있어도 token embedding feature에서는 제외.
        # - action은 별도 `action_sequence`로 처리.
        # -----------------------------------------------
        self.features = [k for k in feature_sequence_dims.keys() if k != "action"]

        # task(target) 이름은 `output_dims`의 key를 그대로 사용
        self.targets = list(output_dims.keys())

        # -----------------------------------------------
        # Feature Embedding Layer
        # - 입력: (batch_size, seq_len)
        # - 출력: (batch_size, seq_len, embedding_dim)
        # -----------------------------------------------
        self.embeddings = nn.ModuleDict(
            {
                feature_name: EmbeddingLayer(int(n_classes), embedding_dim)
                for feature_name, n_classes in feature_sequence_dims.items()
            }
        )

        # -----------------------------------------------
        # Action Weight Lookup (고정 가중치)
        # - action_sequence: (batch_size, seq_len) 형태로 들어오고,
        #   각 timestep의 action id -> weight로 매핑해서 (batch_size, seq_len, 1) weight tensor를 생성.
        # -----------------------------------------------
        self.action_weights = action_weights or {}
        if self.action_weights:
            max_action_id = max(int(k) for k in self.action_weights.keys())
            lookup = torch.ones(max_action_id + 1, dtype=torch.float32)
            for k, v in self.action_weights.items():
                lookup[int(k)] = float(v)
            # register_buffer: to(device) 시 같이 이동, optimizer 업데이트는 하지 않음.
            self.register_buffer("action_lookup", lookup)
        else:
            self.action_lookup = None

        # -----------------------------------------------
        # Positional Encoding
        # -----------------------------------------------
        self.position_encoding = LearnablePositionalEncoding(
            dim_model=embedding_dim, max_len=seq_len
        )

        # -----------------------------------------------
        # Shared Transformer Encoder
        # - 모든 feature를 합친 `x_embed`를 한 번에 인코딩.
        # - behavior sequence transformer처럼 공유된 sequence representation을 생성.
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
        # Experts (Transformer-based)
        # - expert 입력: `shared_seq` (batch_size, seq_len, embedding_dim)
        # - expert 출력: `expert_out` (batch_size, seq_len, embedding_dim)
        # - expert_stack: (batch_size, seq_len, n_experts, embedding_dim) 로 쌓아서 gating으로 mixture 생성.
        # -----------------------------------------------
        self.experts = nn.ModuleList(
            [
                TransformerExpert(
                    embedding_dim=embedding_dim,
                    nhead=self.expert_nhead,
                    num_layers=self.expert_num_layers,
                    dropout=dropout,
                )
                for _ in range(n_experts)
            ]
        )

        # -----------------------------------------------
        # Task-wise Gates
        # - gate 입력은 `gate_input`이 "timestep"냐, "sequence"이냐에 따라 달라짐.
        # - "timestep"이면 `shared_seq`를 사용하고, "sequence"면 `sequence_vector`를 사용함.
        # -----------------------------------------------
        self.gates = nn.ModuleDict(
            {
                target_name: nn.Linear(embedding_dim, n_experts)
                for target_name in self.targets
            }
        )

        # -----------------------------------------------
        # Task-specific Towers (adapter + shallow head)
        #  - adapter: (E -> r -> E) residual 미세조정
        #  - head: layernorm + MLP로 logits 생성
        # -----------------------------------------------
        adapter_rank = max(16, self.embedding_dim // 8)
        head_hidden = max(self.embedding_dim // 2, 32)
        self.towers = nn.ModuleDict()
        for target_name, n_classes in output_dims.items():
            adapter = nn.Sequential(
                nn.Linear(self.embedding_dim, adapter_rank),
                nn.GELU(),
                nn.Linear(adapter_rank, self.embedding_dim),
            )
            head = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, int(n_classes)),
            )
            self.towers[target_name] = nn.ModuleDict({"adapter": adapter, "head": head})

        # -----------------------------------------------
        # Adapter initialization & scaling
        # - adapter를 residual로 더할 때 폭주하지 않도록,
        # - up projection을 매우 작은 std로 초기화하는 패턴.
        # -----------------------------------------------
        for target_name, module in self.towers.items():
            adapter = module["adapter"]
            # adapter = [Linear(D->r), GELU, Linear(r->D)]

            # down projection: Xavier로 기본 안정화
            if isinstance(adapter[0], nn.Linear):
                nn.init.xavier_uniform_(adapter[0].weight)
                if adapter[0].bias is not None:
                    nn.init.zeros_(adapter[0].bias)

            # up projection: std를 매우 작게 -> 초기에는 거의 0에 가까운 residual
            if isinstance(adapter[-1], nn.Linear):
                nn.init.normal_(adapter[-1].weight, std=1e-3)
                if adapter[-1].bias is not None:
                    nn.init.zeros_(adapter[-1].bias)

        # -----------------------------------------------
        # Attention Pooling (옵션)
        # -----------------------------------------------
        if self.global_pool == "att":
            self.attention_pooling = AttentionPooling(hidden_dim=embedding_dim)

        # -----------------------------------------------
        # Causal Attention Mask (look-ahead 차단)
        # -----------------------------------------------
        # causal_attn_mask[t, u] = True if u > t (미래 차단)
        # causal_attn_mask[t, u] = False if u <= t (과거 & 현재 허용)
        causal_bool = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_attn_mask", causal_bool)

        # -----------------------------------------------
        # Non-sequential features
        # -----------------------------------------------
        # 1) sparse: 각 feature -> (batch_size, embedding_dim) 임베딩 후 합산
        self.feature_sparse_dims = feature_sparse_dims or {}
        self.sparse_embeddings = nn.ModuleDict(
            {
                feature_name: EmbeddingLayer(int(dim), self.embedding_dim)
                for feature_name, dim in self.feature_sparse_dims.items()
            }
        )
        # ctx_embed: context vector
        self.sparse_ctx_embed_ln = nn.LayerNorm(self.embedding_dim)

        # 2) dense: concat 후 projector로 (batch_size, embedding) 출력
        self.feature_dense_dims = feature_dense_dims or {}
        self.dense_total_dim = (
            # dense feature에는 스칼라 혹은 벡터 모두 들어올 수 있으므로 전체 차원을 먼저 확인하도록 의도함.
            int(sum(self.feature_dense_dims.values()))
            if self.feature_dense_dims
            else 0
        )
        if self.dense_total_dim > 0:
            hidden = max(int(dense_hidden), self.embedding_dim)
            self.dense_projector = nn.Sequential(
                nn.LayerNorm(self.dense_total_dim),
                nn.Linear(self.dense_total_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embedding_dim),
            )
        else:
            self.dense_projector = None

        # 최종 ctx_embed 스케일 정리
        self.nonseq_ctx_embed_ln = nn.LayerNorm(self.embedding_dim)

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

        # lookup이 없으면 전부 1.0 고정.
        if not hasattr(self, "action_lookup") or self.action_lookup is None:
            return torch.ones(
                (action_sequence.size(0), action_sequence.size(1), 1),
                device=action_sequence.device,
                dtype=dtype,
            )

        # 로컬 복사본을 action_sequence와 같은 device로 맞춤.
        action_lookup = self.action_lookup
        if action_lookup.device != action_sequence.device:
            action_lookup = action_lookup.to(action_sequence.device)

        # action id가 lookup 범위를 넘으면 로컬로 확장. (forward 내에서만)
        max_id_in_batch = int(action_sequence.max().item())
        if max_id_in_batch >= action_lookup.size(0):
            new_lookup = torch.ones(
                max_id_in_batch + 1,
                device=action_sequence.device,
                dtype=action_lookup.dtype,
            )
            new_lookup[: action_lookup.size(0)] = action_lookup.to(
                action_sequence.device
            )
            action_lookup = new_lookup

        # weights: (batch_size, seq_len, 1)
        weights = action_lookup[action_sequence.long()].to(dtype=dtype).unsqueeze(-1)

        # padding 위치는 embedding 곱에서 왜곡 방지용으로 1.0 고정.
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
            # 각 배치의 마지막 유효 토큰 인덱스 계산 (batch_size,)
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
        feature_dense: Dict[str, torch.Tensor] = None,
        feature_sparse: Dict[str, torch.Tensor] = None,
        action_sequence: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """
        :param feature_sequence: {feature_name: 시퀀스 텐서}
        :param feature_dense: {feature_name: Dense feature 텐서}
        :param feature_sparse: {feature_name: Sparse feature 텐서}
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

        # -----------------------------------------------
        # Causal Attention Mask
        # - hared_transformer, expert_transformer 모두 동일한 (seq_len, seq_len) causal mask 사용
        # - causal_attn_mask: (seq_len, seq_len)
        # -----------------------------------------------
        causal_attn_mask = self.causal_attn_mask.to(device=masks.device)

        # -----------------------------------------------
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

        # -----------------------------------------------
        # Shared embedding (feature embedding 합산)
        # -----------------------------------------------
        # x_embed: (batch_size, seq_len, embedding_dim)
        x_embed = 0.0
        for feature_name in self.features:
            # token embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            f_embed = self.embeddings[feature_name](feature_sequence[feature_name])
            x_embed = x_embed + f_embed

        # -----------------------------------------------
        # Action Embedding 추가 (선택)
        # - `action`을 `feature_sequence`로 넣어둔 경우,
        # - `action` 자체를 하나의 token feature처럼 임베딩해서 더해줄 수 있음.
        # - 단, `action_weight`는 따로 곱해지므로 과하게 중복될 수 있어 실험적으로 판단 필요.
        # -----------------------------------------------
        if "action" in feature_sequence and "action" in self.embeddings:
            action_embed = self.embeddings["action"](feature_sequence["action"])
            x_embed = x_embed + action_embed

        # -----------------------------------------------
        # Non-sequential feature
        # - sparse: embedding sum -> (batch_size, embedding_dim)
        # - dense: concat -> projector -> (batch_size, embedding_dim)
        # - ctx_embed는 모든 timestep에 broadcast 합산
        # -----------------------------------------------
        ctx_embed = None

        # 1) sparse
        if feature_sparse:
            sparse_sum = 0.0
            for name, x in feature_sparse.items():
                # x: (batch_size,) or (batch_size, 1)
                if x.dim() > 1:
                    x = x.squeeze(-1)
                if name not in self.sparse_embeddings:
                    continue
                # sparse_sum: (batch_size, embedding_dim)
                sparse_sum = sparse_sum + self.sparse_embeddings[name](x)
            sparse_ctx_embed = self.sparse_ctx_embed_ln(sparse_sum)
            ctx_embed = (
                sparse_ctx_embed
                if ctx_embed is None
                else (ctx_embed + sparse_ctx_embed)
            )

        # 2) dense
        if (
            feature_dense
            and self.dense_projector is not None
            and self.dense_total_dim > 0
        ):
            dense_parts = []
            for name in self.feature_dense_dims.keys():
                if feature_dense is None or name not in feature_dense:
                    continue
                x = feature_dense[name]
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                dense_parts.append(x.float())
            # dense_concat: # (batch_size, dense_total_dim)
            dense_concat = torch.cat(dense_parts, dim=-1)

            # dense_ctx_embed: (batch_size, embedding_dim)
            dense_ctx_embed = self.dense_projector(dense_concat)
            ctx_embed = (
                dense_ctx_embed if ctx_embed is None else (ctx_embed + dense_ctx_embed)
            )

        if ctx_embed is not None:
            # ctx_embed: (batch_size, embedding_dim)
            ctx_embed = self.nonseq_ctx_embed_ln(ctx_embed)
            # x_embed: (batch_size, seq_len, embedding_dim)
            x_embed = x_embed + ctx_embed.unsqueeze(1)

        # -----------------------------------------------
        # Action Weight 적용 (합산된 임베딩에 곱셈)
        # - 합산된 embedding 전체에 timestep-wise 가중치 적용
        # - weights: (batch_size, seq_len, 1) -> broadcasting으로 (batch_size, seq_len, embedding_dim)에 곱해짐
        # -----------------------------------------------
        if weights is not None:
            x_embed = x_embed * weights.to(dtype=x_embed.dtype)

        # positional encoding: (batch_size, seq_len, embedding_dim)
        x_embed = self.position_encoding(x_embed)

        # -----------------------------------------------
        # Shared Transformer Encoder 적용 (미래 토큰 차단)
        # -----------------------------------------------
        # shared_seq: (batch_size, seq_len, embedding_dim)
        shared_seq = self.shared_transformer_encoder(
            x_embed, mask=causal_attn_mask, src_key_padding_mask=masks
        )

        # -----------------------------------------------
        # pooling -> sequence representation
        # -----------------------------------------------
        # sequence_vector: (batch_size, embedding_dim)
        sequence_vector = self._apply_pooling(shared_seq, masks)

        # -----------------------------------------------
        # experts (TransformerExpert)
        # -----------------------------------------------
        # 1) 먼저 각 target의 gate_logits (timestep-wise) 계산
        #    - top_k == n_experts인 경우: top-k 인덱스 계산 및 마스킹을 생략
        #    - top_k < n_experts인 경우: top-k 인덱스를 저장해 이후 마스킹에 사용
        gate_logits_per_target = {}

        use_topk = self.top_k < self.n_experts
        topk_idx_per_target = {} if use_topk else None

        # gate input 선택
        if self.gate_input == "timestep":
            # (batch_size, seq_len, embedding_dim)
            gate_in = self.gate_ln(shared_seq)
        else:
            # (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)로 만들어서 이후 로직(shape)을 통일
            gate_in = self.gate_ln(sequence_vector).unsqueeze(1)

        for target_name in self.targets:

            # gate_logits:
            # - "timestep" 적용 시 (batch_size, seq_len, n_experts)
            # - "sequence" 적용 시 (batch_size, 1, n_experts)
            gate_logits = self.gates[target_name](gate_in)

            # padding 위치는 gating 대상에서 제외 ("timestep" 적용 시만)
            if self.gate_input == "timestep":
                # pad_mask: (batch_size, seq_len, 1)
                pad_mask = masks.unsqueeze(-1)

                # dtype-safe -inf
                neg_inf = -1e9
                if gate_logits.dtype == torch.float16:
                    neg_inf = -torch.finfo(gate_logits.dtype).max / 2.0
                gate_logits = gate_logits.masked_fill(pad_mask, neg_inf)

            gate_logits_per_target[target_name] = gate_logits

            # top_k를 사용할 때만 top-k expert 선택
            if use_topk:
                # idx:
                # "timestep" 적용 시 (batch_size, seq_len, k)
                # "sequence" 적용 시 (batch_size, 1, k)
                _, idx = gate_logits.topk(self.top_k, dim=-1)
                topk_idx_per_target[target_name] = idx

        # -------------------------------------------------
        # (옵션) top-k union만 expert 계산하는 방식
        # - 아래 블록은 "필요한 expert만 계산"하려는 시도인데, 현재는 사용하지 않고 현재는 "모든 expert 계산"을 사용 중임.
        # - 실제로 이 방식은 구현 복잡도가 올라가므로 먼저 전체 expert 계산이 안정적인지 확인 후 시도해도 됨.
        # -------------------------------------------------
        # # 2.1) 모든 타겟·타임스텝에서 필요한 expert들의 union 계산
        # all_idx = torch.cat(
        #     [v.reshape(-1) for v in topk_idx_per_target.values()], dim=0
        # )
        # unique_experts = torch.unique(all_idx).tolist() if all_idx.numel() > 0 else []
        #
        # # 2.2) 필요한 expert들만 계산 (others -> zeros)
        # expert_outputs = [None] * self.n_experts
        # for i in unique_experts:
        #     # TransformerExpert 호출 (shared_seq, mask 적용)
        #     # expert_out: (batch_size, seq_len, embedding_dim)
        #     expert_out = self.experts[i](
        #         shared_seq, src_key_padding_mask=masks, attn_mask=causal_attn_mask
        #     )
        #     expert_outputs[i] = expert_out
        #
        # # 2.3) 계산되지 않은 expert 채우기 (zeros)
        # for i in range(self.n_experts):
        #     if expert_outputs[i] is None:
        #         expert_outputs[i] = torch.zeros_like(shared_seq)

        # 2) 모든 expert를 한 번에 계산
        expert_outputs = []
        for expert in self.experts:
            # out: (batch_size, seq_len, embedding_dim)
            out = expert(
                shared_seq, src_key_padding_mask=masks, attn_mask=causal_attn_mask
            )
            expert_outputs.append(out)

        # 3) expert_stack 생성
        # - expert_stack: (batch_size, seq_len, n_experts, embedding_dim)
        expert_stack = torch.stack(expert_outputs, dim=2)

        # -----------------------------------------------
        # task-wise routing + prediction
        # -----------------------------------------------
        y_outputs: Dict[str, torch.Tensor] = {}

        # -----------------------------------------------
        # 각 timestep마다 서로 다른 expert 조합 선택을 위해 timestep-wise 계산
        # -----------------------------------------------
        for target_name in self.targets:
            gate_logits = gate_logits_per_target[target_name]

            # 4) temperature softmax 준비
            # temp = self.gate_temp_min + (
            #     self.gate_temp_max - self.gate_temp_min
            # ) * torch.sigmoid(self.gate_temperature)
            temp = 1

            if use_topk:
                # sparse routing: top-k만 남기고 나머지는 -inf
                topk_idx = topk_idx_per_target[target_name]

                # 1) top-k mask 생성
                # topk_mask: (batch_size, seq_len, n_experts)
                topk_mask = torch.zeros_like(gate_logits, dtype=torch.bool)
                topk_mask = topk_mask.scatter_(-1, topk_idx, True)

                # 2) padding 위치는 gating 대상에서 제외 (단, "timestep" 적용 시에만)
                if self.gate_input == "timestep":
                    # masks: (batch_size, seq_len) -> (batch_size, seq_len, 1)
                    pad_mask = masks.unsqueeze(-1)
                    topk_mask = topk_mask & (~pad_mask)

                # 3) top-k가 아닌 expert는 -inf로 보내 softmax 확률이 0에 수렴하게 함
                # - 이렇게 하면 top-k만 섞는 sparse routing이 강제됨
                neg_inf = -1e9
                if gate_logits.dtype == torch.float16:
                    # fp16에서 -1e9가 overflow될 수 있어 dtype 기반으로 안전하게 처리
                    neg_inf = -torch.finfo(gate_logits.dtype).max / 2.0
                sparse_logits = gate_logits.masked_fill(~topk_mask, neg_inf)

                gate_weights = torch.softmax(sparse_logits / temp, dim=-1)
            else:
                # 모든 expert를 그대로 사용 (top-k 마스킹 생략)
                gate_weights = torch.softmax(gate_logits / temp, dim=-1)

            # gate_weights:
            # "timestep" -> (batch_size, seq_len, n_experts)
            # "sequence" -> (batch_size, 1, n_experts)

            if self.gate_input == "sequence":
                # "sequence" 적용 시 (batch_size, seq_len, n_experts)로 broadcast 필요
                gate_weights = gate_weights.expand(-1, expert_stack.size(1), -1)

            # 5) fused_seq = Σ_k gate_weight_k * expert_out_k
            # gate_expand: (batch_size, seq_len, n_experts, 1)
            gate_expand = gate_weights.unsqueeze(-1)
            # fused_seq: (batch_size, seq_len, embedding_dim)
            fused_seq = (expert_stack * gate_expand).sum(dim=2)

            # 6) target별 adapter + head로 logits 생성
            tower = self.towers[target_name]
            adapter = tower["adapter"]
            head = tower["head"]

            # adapter는 residual 형태로 적용
            # adapted: (batch_size, seq_len, embedding_dim)
            adapted = adapter(fused_seq)
            # fused_res: (batch_size, seq_len, embedding_dim)
            fused_res = fused_seq + adapted

            # head로 logits 생성
            # logits: (batch_size, seq_len, n_classes)
            logits = head(fused_res)
            y_outputs[target_name] = logits

        return {"sequence_vector": sequence_vector, "y_outputs": y_outputs}
