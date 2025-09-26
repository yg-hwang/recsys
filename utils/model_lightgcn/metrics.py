import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Generator, Any


def compute_bpr_loss(
    users, users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0
) -> Tuple[torch.Tensor, torch.Tensor]:
    reg_loss = (
        (1 / 2)
        * (user_emb_0.norm().pow(2) + pos_emb_0.norm().pow(2) + neg_emb_0.norm().pow(2))
        / float(len(users))
    )
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)

    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    return loss, reg_loss


def get_hit_list(item_idx, top_relevant_item) -> List[int]:
    return [1 if x in set(item_idx) else 0 for x in top_relevant_item]


def get_dcg_idcg(item_idx, hit_list) -> float:
    idcg = sum(
        [1 / np.log1p(idx + 1) for idx in range(min(len(item_idx), len(hit_list)))]
    )
    dcg = sum([hit / np.log1p(idx + 1) for idx, hit in enumerate(hit_list)])
    return dcg / idcg


def get_cumsum(hit_list) -> np.ndarray:
    return np.cumsum(hit_list)


def get_map(item_idx, hit_list, hit_list_cumsum) -> float:
    return sum(
        [
            hit_cumsum * hit / (idx + 1)
            for idx, (hit, hit_cumsum) in enumerate(zip(hit_list, hit_list_cumsum))
        ]
    ) / len(item_idx)


def get_metrics(
    user_embed_wts: torch.Tensor,
    item_embed_wts: torch.Tensor,
    test_data: pd.DataFrame,
    top_k: int,
    metrics: list = None,
) -> Generator[float, Any, None]:
    if metrics is None:
        metrics = ["recall", "precision", "ndcg", "map"]

    relevance_score = torch.matmul(
        user_embed_wts, torch.transpose(item_embed_wts, 0, 1)
    )

    # calculating top K metrics
    top_k_relevance_indices = torch.topk(relevance_score, top_k).indices

    df_top_k_relevance_indices = pd.DataFrame(
        top_k_relevance_indices.numpy(),
        columns=["top_idx_" + str(x + 1) for x in range(top_k)],
    )

    df_top_k_relevance_indices["user_ID"] = df_top_k_relevance_indices.index
    df_top_k_relevance_indices["top_relevant_item"] = df_top_k_relevance_indices[
        ["top_idx_" + str(x + 1) for x in range(top_k)]
    ].values.tolist()
    df_top_k_relevance_indices = df_top_k_relevance_indices[
        ["user_ID", "top_relevant_item"]
    ]

    test_interacted_items = (
        test_data.groupby("user_idx")["item_idx"].apply(list).reset_index()
    )

    df_metrics = pd.merge(
        test_interacted_items,
        df_top_k_relevance_indices,
        how="left",
        left_on="user_idx",
        right_on=["user_ID"],
    )
    df_metrics["interaction_item"] = [
        list(set(a).intersection(b))
        for a, b in zip(df_metrics.item_idx, df_metrics.top_relevant_item)
    ]

    if "recall" in metrics:
        df_metrics["recall"] = df_metrics.apply(
            lambda x: len(x["interaction_item"]) / len(x["item_idx"]), axis=1
        )

    if "precision" in metrics:
        df_metrics["precision"] = df_metrics.apply(
            lambda x: len(x["interaction_item"]) / top_k, axis=1
        )

    if "ndcg" in metrics:
        df_metrics["hit_list"] = df_metrics.apply(
            lambda x: get_hit_list(x["item_idx"], x["top_relevant_item"]), axis=1
        )
        df_metrics["ndcg"] = df_metrics.apply(
            lambda x: get_dcg_idcg(x["item_idx"], x["hit_list"]), axis=1
        )

    if "map" in metrics:
        df_metrics["hit_list_cumsum"] = df_metrics.apply(
            lambda x: get_cumsum(x["hit_list"]), axis=1
        )

        df_metrics["map"] = df_metrics.apply(
            lambda x: get_map(x["item_idx"], x["hit_list"], x["hit_list_cumsum"]),
            axis=1,
        )

    return (df_metrics[m].mean() for m in metrics)
