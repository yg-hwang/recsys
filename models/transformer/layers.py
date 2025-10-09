import torch
import torch.nn as nn


class EmbeddingWithNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.embedding(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_len: int = 10):
        """
        위치 임베딩(Positional Encoding) 클래스
        - Transformer는 RNN처럼 순차적 구조가 없으므로,
          입력 토큰의 순서를 알 수 있도록 sin, cos 함수를 이용한 위치 정보를 추가하기 위한 기능

        :param dim_model: 임베딩 차원 크기
        :param max_len: 시퀀스 최대 길이
        """
        super(PositionalEncoding, self).__init__()

        # -----------------------------------------------
        # 위치 벡터 (0, 1, 2, ..., max_len-1)
        # -----------------------------------------------
        # 각 row는 시퀀스 내 위치를 의미
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        # -----------------------------------------------
        # 주기(term) 계산
        # -----------------------------------------------
        # Transformer 논문 공식:
        # PE(pos, 2i)   = sin(pos / 10000^(2i/dim_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/dim_model))
        div_term = torch.exp(
            torch.arange(0, dim_model, 2)  # 0, 2, 4, ..., (dim_model-2)
            * (-torch.log(torch.tensor(10000.0)) / dim_model)
        )  # (dim_model // 2)

        # -----------------------------------------------
        # 위치 임베딩 행렬 초기화
        # -----------------------------------------------
        pe = torch.zeros(max_len, dim_model)

        # 짝수 index (2i): sin 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)

        # 홀수 index (2i+1): cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)

        # -----------------------------------------------
        # 배치 브로드캐스팅을 위해 차원 추가
        # -----------------------------------------------
        # (max_len, 1, dim_model)
        # - seq_len: max_len
        # - batch_size: 1 (추후 broadcast)
        # - embedding_dim: dim_model
        pe = pe.unsqueeze(1)

        # -----------------------------------------------
        # 학습 파라미터가 아닌 buffer로 등록
        # -----------------------------------------------
        # - 학습 중 업데이트 되지 않음
        # - GPU/CPU 전환 시 자동 이동
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (입력에 위치 임베딩 더하기)

        :param x: 입력 텐서 (seq_len, batch_size, embedding_dim)
        :return: 위치 인코딩이 더해진 텐서 (seq_len, batch_size, embedding_dim)
        """
        # 현재 입력 길이(seq_len)에 해당하는 위치 임베딩 잘라서 더해줌
        # Broadcasting: (seq_len, batch_size, dim) + (seq_len, 1, dim)
        # Position encoding은 입력 x에 더해줌 (broadcast)
        x = x + self.pe[: x.size(0), :]

        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_len: int = 20, dropout: float = 0.0):
        """
        학습 가능한 (learnable) 위치 임베딩 클래스
        - 기존 sin/cos 기반 absolute encoding 대신 학습 가능한 embedding 사용
        - 모델이 직접 위치별 중요도를 학습하게 함

        :param dim_model: 임베딩 차원 크기
        :param max_len: 시퀀스 최대 길이
        :param dropout: 위치 인코딩 추가 후 dropout 비율 (optional)
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, dim_model)
        self.dropout = nn.Dropout(p=dropout)

        # 파라미터 초기화 (Transformer 논문 방식)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 입력 텐서 (seq_len, batch_size, embedding_dim)
        :return: 위치 인코딩이 더해진 텐서 (seq_len, batch_size, embedding_dim)
        """
        seq_len, batch_size, embedding_dim = x.size()

        # 각 위치 인덱스 생성: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        # 위치 embedding lookup 후 broadcast (seq_len, embedding_dim)
        pe = self.pos_embedding(positions)

        # 입력 x에 더하고 dropout
        x = x + pe

        return self.dropout(x)
