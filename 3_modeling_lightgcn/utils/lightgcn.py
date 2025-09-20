import random
import torch
import numpy as np
import pandas as pd
import polars as pl
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Tuple, Union
from scipy.sparse import coo_matrix, dok_matrix, diags
from sklearn.preprocessing import LabelEncoder


class LightGCN(nn.Module):
    def __init__(self, data: pd.DataFrame, n_layers: int, latent_dim: int):
        super(LightGCN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.embed_init = None
        self._create_idx_map(data)
        self._create_interaction(data)
        self._init_embedding()
        self.norm_adj_mat_sparse_tensor = self.get_diffusion_matrix(data=data)

    def _create_idx_map(self, data: Union[pd.DataFrame, pl.DataFrame]) -> None:
        encoder_user = LabelEncoder()
        encoder_item = LabelEncoder()
        data["user_idx"] = encoder_user.fit_transform(data["user_id"].to_numpy())
        data["item_idx"] = encoder_item.fit_transform(data["item_id"].to_numpy())

        self.user_map = dict(zip(data["user_id"].to_list(), data["user_idx"].to_list()))
        self.item_map = dict(zip(data["item_id"].to_list(), data["item_idx"].to_list()))
        self.n_users = data["user_idx"].nunique()
        self.n_items = data["item_idx"].nunique()
        self.user_idx_list = data["user_idx"].unique().tolist()

    def _create_interaction(self, data: pd.DataFrame) -> None:
        self.interaction = data.groupby("user_idx")["item_idx"].apply(list).to_dict()

    def _init_embedding(self):
        self.embed_init = nn.Embedding(self.n_users + self.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.embed_init.weight)
        self.embed_init.weight = nn.Parameter(self.embed_init.weight)

    @staticmethod
    def _sample_negative_item(item_id_list: list, n_items: int):
        while True:
            neg_id = random.randint(0, n_items - 1)
            if neg_id not in item_id_list:
                return neg_id

    def load_data(self, batch_size: int) -> Tuple:

        if self.n_users < batch_size:
            user_idx_list = [
                random.choice(self.user_idx_list) for _ in range(batch_size)
            ]
        else:
            user_idx_list = random.sample(self.user_idx_list, batch_size)

        user_idx_list.sort()

        pos_items, neg_items = [], []
        for user_idx in user_idx_list:
            item_list = self.interaction[user_idx]
            pos_items.append(random.choice(item_list))
            neg_items.append(
                self._sample_negative_item(item_id_list=item_list, n_items=self.n_items)
            )

        return user_idx_list, pos_items, neg_items

    def get_diffusion_matrix(self, data: pd.DataFrame) -> torch.Tensor:
        user_idx = data["user_idx"].to_numpy()
        item_idx = data["item_idx"].to_numpy()

        # User-Item Interaction Matrix (R)
        R = dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        R[user_idx, item_idx] = 1.0

        # Adjacency Matrix (symmetric bipartite graph)
        adj_mat = dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        )

        # 루프 대신 벡터 연산으로 adjacency 추가
        rows = np.concatenate([user_idx, item_idx + self.n_users])
        cols = np.concatenate([item_idx + self.n_users, user_idx])
        values = np.ones(len(rows), dtype=np.float32)
        adj_mat = coo_matrix((values, (rows, cols)), shape=adj_mat.shape)

        # Symmetric Normalization D^(-1/2) * A * D^(-1/2)
        row_sum = np.array(adj_mat.sum(1)).flatten()
        d_inv = np.power(row_sum + 1e-9, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = diags(d_inv)

        norm_adj_mat = d_mat_inv.dot(adj_mat).dot(d_mat_inv)

        # Convert to PyTorch Sparse Tensor
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

    def propagate_through_layers(self):
        all_layer_embedding = [self.embed_init.weight]
        embedding_weight = self.embed_init.weight

        for layer in range(self.n_layers):
            embedding_weight = torch.sparse.mm(
                self.norm_adj_mat_sparse_tensor, embedding_weight
            ).to(self.device)
            all_layer_embedding.append(embedding_weight)

        all_layer_embedding = torch.stack(all_layer_embedding)
        mean_layer_embedding = torch.mean(all_layer_embedding, axis=0)

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

    def forward(self, users, pos_items, neg_items):
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
        user_emb_0, pos_emb_0, neg_emb_0 = (
            initial_user_embed[users],
            initial_item_embed[pos_items],
            initial_item_embed[neg_items],
        )

        return users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0

    def predict(self, user_id_list):
        final_user_embed, final_item_embed, _, _ = self.propagate_through_layers()
        K = self.n_items
        user_ids = []
        for user_id in user_id_list:
            user_ids.append(self.user_map.get(user_id))

        relevance_score = torch.matmul(
            final_user_embed[user_ids], torch.transpose(final_item_embed, 0, 1)
        )
        top_k_relevance_score = torch.topk(relevance_score, K).values
        top_k_relevance_indices = torch.topk(relevance_score, K).indices

        results = {}
        for user_id, ind, score in zip(
            user_id_list, top_k_relevance_indices, top_k_relevance_score
        ):
            items = [self.item_map[idx] for idx in ind.tolist()]
            results[user_id] = {"items": items, "scores": score.tolist()}

        return results
