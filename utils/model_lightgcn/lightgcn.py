import random
import torch
import numpy as np
import pandas as pd
import polars as pl
import torch.nn as nn
from typing import Tuple, Union
from scipy.sparse import coo_matrix, dok_matrix, diags
from sklearn.preprocessing import LabelEncoder


class LightGCN(nn.Module):
    def __init__(self, data: pd.DataFrame, n_layers: int, latent_dim: int):
        """
        LightGCN 모델 클래스
        - 인접 행렬(유저-아이템 그래프)을 기반으로 전파(propagation)를 수행하여 유저와 아이템 임베딩을 학습하는 그래프 모델

        :param data: 유저-아이템 상호작용 데이터 (user_id, item_id 포함)
        :param n_layers: 그래프 전파 레이어 수 (확산 횟수)
        :param latent_dim: 임베딩 차원 수
        """

        super(LightGCN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.embed_init = None

        # 유저, 아이템 ID를 0 ~ N 범위의 연속 정수 인덱스로 변환
        self._create_idx_map(data)

        # 유저별 아이템 상호작용 딕셔너리 생성
        self._create_interaction(data)

        # 유저 + 아이템 임베딩 테이블 초기화
        self._init_embedding()

        # 그래프 확산 행렬 (정규화 adjacency matrix) 계산 및 sparse tensor 저장
        self.norm_adj_mat_sparse_tensor = self.get_diffusion_matrix(data=data)

    # -----------------------------------------------
    # 전처리 모듈
    # -----------------------------------------------
    def _create_idx_map(self, data: Union[pd.DataFrame, pl.DataFrame]) -> None:
        """
        유저와 아이템 ID를 정수 인덱스로 매핑
        - PyTorch Embedding은 0~N-1 인덱스만 받으므로 변환 필요

        :param data: 유저-아이템 상호작용 데이터
        :return: None (클래스 내부에 user_map, item_map, n_users, n_items 저장)
        """

        encoder_user = LabelEncoder()
        encoder_item = LabelEncoder()

        # user_id, item_id -> user_idx, item_idx
        data["user_idx"] = encoder_user.fit_transform(data["user_id"].to_numpy())
        data["item_idx"] = encoder_item.fit_transform(data["item_id"].to_numpy())

        # 매핑 딕셔너리 저장 (id -> index)
        self.user_map = dict(zip(data["user_id"].to_list(), data["user_idx"].to_list()))
        self.item_map = dict(zip(data["item_id"].to_list(), data["item_idx"].to_list()))

        # 유저, 아이템 개수
        self.n_users = data["user_idx"].nunique()
        self.n_items = data["item_idx"].nunique()

        # 학습 시 샘플링할 유저 인덱스 목록
        self.user_idx_list = data["user_idx"].unique().tolist()

    # -----------------------------------------------
    # 전처리 모듈
    # -----------------------------------------------
    def _create_interaction(self, data: pd.DataFrame) -> None:
        """
        유저별 상호작용 아이템 리스트 생성
        - 네거티브 샘플링 시 "이 유저가 보지 않은 아이템"을 찾을 때 활용

        :param data: 유저-아이템 상호작용 데이터 (user_idx, item_idx 포함)
        :return: None (클래스 내부에 interaction 딕셔너리 저장)
        """

        self.interaction = data.groupby("user_idx")["item_idx"].apply(list).to_dict()

    def _init_embedding(self) -> None:
        """
        유저, 아이템 임베딩 초기화
        - (U + I, latent_dim) 크기의 임베딩 테이블 생성
        - Xavier Uniform 분포로 초기화 (딥러닝 임베딩 초기화 기본값 중 하나)

        :return: None (embed_init에 임베딩 저장)
        """

        self.embed_init = nn.Embedding(self.n_users + self.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.embed_init.weight)
        self.embed_init.weight = nn.Parameter(self.embed_init.weight)

    # -----------------------------------------------
    # 학습 샘플링 (BPR Loss용)
    # -----------------------------------------------
    @staticmethod
    def _sample_negative_item(item_id_list: list, n_items: int) -> int:
        """
        아이템 네거티브 샘플링 (유저가 보지 않은 아이템을 무작위 선택)

        :param item_id_list: 해당 유저가 상호작용한 아이템 리스트
        :param n_items: 전체 아이템 개수
        :return: 샘플링된 negative item index
        """

        while True:
            neg_id = random.randint(0, n_items - 1)
            if neg_id not in item_id_list:
                return neg_id

    def load_data(self, batch_size: int) -> Tuple[list, list, list]:
        """
        학습용 배치 데이터 로드 (user, positive item, negative item)

        :param batch_size: 배치 크기
        :return: (user_idx_list, pos_items, neg_items)
        """

        # 유저 배치 선택 (유저 수 < batch_size면 중복 허용)
        if self.n_users < batch_size:
            user_idx_list = [
                random.choice(self.user_idx_list) for _ in range(batch_size)
            ]
        else:
            user_idx_list = random.sample(self.user_idx_list, batch_size)

        # 정렬해줘야 같은 순서로 매칭 가능
        user_idx_list.sort()

        pos_items, neg_items = [], []
        for user_idx in user_idx_list:
            # 유저가 본 아이템들
            item_list = self.interaction[user_idx]
            # Positive 아이템 샘플
            pos_items.append(random.choice(item_list))
            # Negative 아이템 샘플
            neg_items.append(self._sample_negative_item(item_list, self.n_items))

        return user_idx_list, pos_items, neg_items

    # -----------------------------------------------
    # 그래프 확산 행렬 생성
    # -----------------------------------------------
    def get_diffusion_matrix(self, data: pd.DataFrame) -> torch.Tensor:
        """
        LightGCN 그래프 확산 행렬 (정규화된 adjacency matrix) 생성

        :param data: 유저-아이템 상호작용 데이터 (user_idx, item_idx 포함)
        :return: 정규화 adjacency matrix (PyTorch sparse tensor)
        """

        user_idx = data["user_idx"].to_numpy()  # 예: [0, 0, 1, 2, 2, ...]
        item_idx = data["item_idx"].to_numpy()  # 예: [3, 7, 2, 1, 9, ...]

        # 전체 그래프 인접행렬 크기: (U+I) x (U+I)
        # - U: 유저 수, I: 아이템 수
        adj_mat = dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        )

        # user-node와 item-node 연결
        # - 유저와 아이템을 한 행렬 안에서 구분하기 위해 아이템 노드에는 유저 수 만큼의 offset 추가)
        # - 예: user=0, item=2 -> 실제 위치는 (row=0, col=U+2)
        rows = np.concatenate([user_idx, item_idx + self.n_users])
        cols = np.concatenate([item_idx + self.n_users, user_idx])

        # edge(연결)는 모두 weight=1로 설정
        values = np.ones(len(rows), dtype=np.float32)

        # COO 형식(좌표 리스트 방식)으로 희소 행렬로 변환
        adj_mat = coo_matrix((values, (rows, cols)), shape=adj_mat.shape)

        # 각 노드별 연결 개수(차수, degree) 계산
        row_sum = np.array(adj_mat.sum(1)).flatten()

        # degree의 역제곱근을 계산: D^{-1/2}
        # (연결이 많은 노드의 영향력이 너무 커지지 않도록 조절)
        d_inv = np.power(row_sum + 1e-9, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = diags(d_inv)

        # 최종 정규화된 인접행렬: D^{-1/2} * A * D^{-1/2}
        norm_adj_mat = d_mat_inv.dot(adj_mat).dot(d_mat_inv)

        # PyTorch sparse 텐서(coo 형식)로 변환
        norm_adj_mat_coo = norm_adj_mat.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((norm_adj_mat_coo.row, norm_adj_mat_coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(norm_adj_mat_coo.data)
        shape = torch.Size(norm_adj_mat_coo.shape)

        norm_adj_mat_sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, device=self.device
        )

        return norm_adj_mat_sparse_tensor

    # -----------------------------------------------
    # GCN 전파
    # -----------------------------------------------
    def propagate_through_layers(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LightGCN 레이어 전파 (여러 레이어를 통과하여 최종 임베딩 계산)
        - 초기 임베딩을 정규화 adjacency matrix와 곱하여 확산
        - 모든 레이어의 임베딩을 평균 내어 최종 임베딩 생성

        :return: (final_user_embed, final_item_embed, initial_user_embed, initial_item_embed)
        """

        all_layer_embedding = [self.embed_init.weight]
        embedding_weight = self.embed_init.weight

        for layer in range(self.n_layers):
            embedding_weight = torch.sparse.mm(
                self.norm_adj_mat_sparse_tensor, embedding_weight
            ).to(self.device)
            all_layer_embedding.append(embedding_weight)

        # 모든 레이어 임베딩 평균
        all_layer_embedding = torch.stack(all_layer_embedding)
        mean_layer_embedding = torch.mean(all_layer_embedding, axis=0)

        # 유저, 아이템 임베딩 분리
        final_user_embed, final_item_embed = torch.split(
            tensor=mean_layer_embedding,
            split_size_or_sections=[self.n_users, self.n_items],
        )
        initial_user_embed, initial_item_embed = torch.split(
            tensor=self.embed_init.weight,
            split_size_or_sections=[self.n_users, self.n_items],
        )

        return (
            final_user_embed,
            final_item_embed,
            initial_user_embed,
            initial_item_embed,
        )

    # -----------------------------------------------
    # Forward & 예측
    # -----------------------------------------------
    def forward(self, users, pos_items, neg_items) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward propagation
        - 유저, positive item, negative item의 임베딩 반환
        - BPR Loss 계산 시 활용

        :param users: 유저 인덱스 리스트
        :param pos_items: Positive 아이템 인덱스 리스트
        :param neg_items: Negative 아이템 인덱스 리스트
        :return: (users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0)
        """

        # 최종 임베딩
        (
            final_user_embed,
            final_item_embed,
            initial_user_embed,
            initial_item_embed,
        ) = self.propagate_through_layers()

        users_emb, pos_emb, neg_emb = (
            final_user_embed[users],
            final_item_embed[pos_items],
            final_item_embed[neg_items],
        )

        # 초기 임베딩 (regularization용)
        user_emb_0, pos_emb_0, neg_emb_0 = (
            initial_user_embed[users],
            initial_item_embed[pos_items],
            initial_item_embed[neg_items],
        )

        return users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0

    def predict(self, user_id_list: list) -> dict:
        """
        주어진 유저 ID 리스트에 대해 아이템 추천 수행

        주어진 유저 ID 리스트에 대해 아이템 추천 수행
        - 각 유저 임베딩과 아이템 임베딩 간 내적 점수 계산
        - 점수 상위 K개 아이템 반환 (단순 실험 용도로만 사용하고, 실제 서비스에선 ANN을 빌드하여 사용 권장)

        :param user_id_list: 추천을 받을 유저 ID 리스트
        :return: {user_id: {"items": 추천 아이템 리스트, "scores": 해당 점수 리스트}}
        """

        final_user_embed, final_item_embed, _, _ = self.propagate_through_layers()
        K = self.n_items

        # user_id를 내부 index로 변환
        user_ids = []
        for user_id in user_id_list:
            user_ids.append(self.user_map.get(user_id))

        # relevance score = user_emb * item_emb^T
        relevance_score = torch.matmul(
            final_user_embed[user_ids], torch.transpose(final_item_embed, 0, 1)
        )

        # 상위 K개 아이템 선택
        top_k_relevance_score = torch.topk(relevance_score, K).values
        top_k_relevance_indices = torch.topk(relevance_score, K).indices

        results = {}
        for user_id, ind, score in zip(
            user_id_list, top_k_relevance_indices, top_k_relevance_score
        ):
            # 내부 index를 원래 item_id로 변환
            items = [self.item_map[idx] for idx in ind.tolist()]
            results[user_id] = {"items": items, "scores": score.tolist()}

        return results
