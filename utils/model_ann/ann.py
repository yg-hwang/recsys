import hnswlib
import numpy as np
from typing import Dict


def build_index(
    vectors: np.ndarray,
    *,
    space: str = "cosine",  # "ip" | "cosine" | "l2"
    M: int = 16,
    ef_construction: int = 128,
    ef: int = 32,
    num_threads: int | None = None,
) -> hnswlib.Index:
    """
    주어진 벡터로 hnswlib 인덱스를 빌드합니다.

    :param vectors: (num, dim) 형태의 float32 배열
    :param space: 유사도/거리 공간 ("ip", "cosine", "l2")
                  - "ip": 내적(점수 클수록 유사)
                  - "cosine": 코사인 거리(작을수록 유사; 필요시 후처리로 1 - d로 유사도 변환)
                  - "l2": L2 거리(작을수록 유사)
    :param M: 그래프의 연결 정도
    :param ef_construction: 빌드 시 탐색 폭
    :param ef: 검색 시 탐색 폭(리콜↑, 속도↓)
    :param num_threads: 검색에 사용할 스레드 수 (None이면 라이브러리 기본값)
    :return: hnswlib Index
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    num, dim = vectors.shape
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=num, M=M, ef_construction=ef_construction)
    # 내부 라벨을 0..num-1로 부여
    labels = np.arange(num, dtype=np.int32)
    index.add_items(vectors, labels)
    index.set_ef(ef)
    if num_threads is not None:
        index.set_num_threads(num_threads)
    return index


def search(
    query_vector: np.ndarray,
    index: hnswlib.Index,
    top_k: int = 5,
) -> Dict[int, float] | list[Dict[int, float]]:
    """
    쿼리 벡터(단일 또는 배치)에 대해 인덱스에서 근접 이웃을 검색합니다.
    반환 형식:
      - 단일 벡터 입력: {내부인덱스: 점수}
      - 다중 벡터 입력: [{내부인덱스: 점수}, ...] (쿼리 순서 동일)

    :param query_vector: (dim,), (1, dim) 또는 (N, dim) 형태의 float32 배열
    :param index: hnswlib 인덱스
    :param top_k: 반환할 이웃 수
    """
    # 입력 형태 정규화
    is_single = query_vector.ndim == 1
    query = query_vector.reshape(1, -1) if is_single else query_vector
    if query.dtype != np.float32:
        query = query.astype(np.float32)

    # hnswlib는 배치 질의를 지원
    labels, distances = index.knn_query(query, k=top_k)

    # 점수 해석:
    # - space=="ip": distances가 내적 점수(클수록 유사) -> 그대로 사용
    # - space=="cosine"/"l2": distances는 거리(작을수록 유사) -> 필요시 후처리에서 변환
    results: list[Dict[int, float]] = []
    for q in range(labels.shape[0]):
        result_q: Dict[int, float] = {}
        for rank, lbl in enumerate(labels[q]):
            if lbl == -1:
                continue
            score = float(distances[q][rank])
            if index.get_current_count() > 0:
                if index.space == "cosine":
                    score = 1 - score
                elif index.space == "l2":
                    score = 1 / (1 + score)
            result_q[int(lbl)] = score
        results.append(result_q)

    return results[0] if is_single else results
