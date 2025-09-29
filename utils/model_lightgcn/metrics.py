import torch
import pandas as pd
from typing import Tuple, Generator, Any


def compute_bpr_loss(
    users, users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BPR Loss 계산 함수

    - Bayesian Personalized Ranking Loss는 추천 모델 학습에서 "positive item은 negative item보다 점수가 높아야 한다"라는 제약을 학습
    - 즉, (u, i+) 쌍이 (u, i-) 쌍보다 relevance score가 높도록 유도

    :param users: 배치 유저 인덱스 리스트
    :param users_emb: 최종 user embedding (전파 후)
    :param pos_emb: 최종 positive item embedding
    :param neg_emb: 최종 negative item embedding
    :param user_emb_0: 초기 user embedding (regularization용)
    :param pos_emb_0: 초기 positive item embedding (regularization용)
    :param neg_emb_0: 초기 negative item embedding (regularization용)
    :return: (BPR loss, 정규화 regularization loss)
    """
    # -----------------------------------------------
    # 정규화 손실 (L2 norm 기반)
    # - overfitting 방지
    # - 초기 임베딩 가중치의 크기를 제한
    # -----------------------------------------------
    reg_loss = (
        (1 / 2)
        * (user_emb_0.norm().pow(2) + pos_emb_0.norm().pow(2) + neg_emb_0.norm().pow(2))
        / float(len(users))
    )

    # -----------------------------------------------
    # positive / negative score 계산
    # - relevance score = 내적 (dot product)
    # -----------------------------------------------
    pos_scores = torch.mul(users_emb, pos_emb)  # element-wise 곱
    pos_scores = torch.sum(pos_scores, dim=1)  # 내적 결과 (배치 크기의 벡터)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)

    # -----------------------------------------------
    # BPR Loss = softplus(neg - pos)
    # - pos > neg 되도록 학습
    # - softplus(x) = log(1 + exp(x)) -> smooth hinge 역할
    # -----------------------------------------------
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    return loss, reg_loss


def get_metrics(
    user_embed_wts: torch.Tensor,
    item_embed_wts: torch.Tensor,
    test_data: pd.DataFrame,
    top_k: int,
    metrics: list = None,
) -> Generator[float, Any, None]:
    """
    추천 모델의 오프라인 평가 지표 계산 함수 (Recall@K, Precision@K)
    - 각 유저별로 학습된 임베딩 기반 Top-K 추천 아이템을 뽑아 실제 test interaction과 비교하여 평가

    :param user_embed_wts: 최종 학습된 user embedding matrix (U x d)
    :param item_embed_wts: 최종 학습된 item embedding matrix (I x d)
    :param test_data: 테스트용 상호작용 데이터 (user_idx, item_idx 포함)
    :param top_k: 추천 상위 K
    :param metrics: 계산할 지표 목록 (["recall", "precision"])
    :return: generator (metrics 순서대로 평균값 yield)
    """

    if metrics is None:
        metrics = ["recall", "precision"]

    # -----------------------------------------------
    # relevance score 계산 (전체 user * 전체 item 내적)
    # -----------------------------------------------
    relevance_score = torch.matmul(
        user_embed_wts, torch.transpose(item_embed_wts, 0, 1)
    )

    # -----------------------------------------------
    # 각 유저별 top-K 아이템 index 추출
    # -----------------------------------------------
    top_k_relevance_indices = torch.topk(relevance_score, top_k).indices

    # Pandas DataFrame으로 변환 (분석 편의용)
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

    # -----------------------------------------------
    # 실제 test interaction 데이터 준비
    # -----------------------------------------------
    test_interacted_items = (
        test_data.groupby("user_idx")["item_idx"].apply(list).reset_index()
    )

    # -----------------------------------------------
    # 추천 결과와 실제 데이터 매칭
    # -----------------------------------------------
    df_metrics = pd.merge(
        test_interacted_items,
        df_top_k_relevance_indices,
        how="left",
        left_on="user_idx",
        right_on=["user_ID"],
    )

    # 교집합 아이템 (실제 본 것 ∩ 추천한 것)
    df_metrics["interaction_item"] = [
        list(set(a).intersection(b))
        for a, b in zip(df_metrics.item_idx, df_metrics.top_relevant_item)
    ]

    # -----------------------------------------------
    # Recall@K 계산
    # - 실제 본 아이템 중 몇 %를 Top-K가 맞췄는가
    # -----------------------------------------------
    if "recall" in metrics:
        df_metrics["recall"] = df_metrics.apply(
            lambda x: len(x["interaction_item"]) / len(x["item_idx"]), axis=1
        )

    # -----------------------------------------------
    # Precision@K 계산
    # - Top-K 추천 중 실제 본 아이템의 비율
    # -----------------------------------------------
    if "precision" in metrics:
        df_metrics["precision"] = df_metrics.apply(
            lambda x: len(x["interaction_item"]) / top_k, axis=1
        )

    # -----------------------------------------------
    # 각 지표 평균값 반환
    # -----------------------------------------------
    return (df_metrics[m].mean() for m in metrics)
