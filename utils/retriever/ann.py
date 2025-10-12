import hnswlib
import numpy as np
from pathlib import Path
from typing import Dict, Union


def build_index(
    vectors: np.ndarray,
    *,
    space: str = "cosine",  # "ip" | "cosine" | "l2"
    M: int = 16,
    ef_construction: int = 128,
    ef: int = 32,
    num_threads: int = None,
    save_path: Union[str, Path] = None,
) -> hnswlib.Index:
    """
    주어진 벡터로 hnswlib 인덱스 빌드

    :param vectors: (num, dim) 형태의 float32 배열
    :param space: 유사도 및 거리 공간 ("ip", "cosine", "l2")
                  - "ip": 내적(점수 클수록 유사)
                  - "cosine": 코사인 거리(작을수록 유사) -> 필요시 후처리로 (1 - d)로 유사도 변환
                  - "l2": L2 거리(작을수록 유사)
    :param M: 그래프의 연결 정도
    :param ef_construction: 빌드 시 탐색 폭
    :param ef: 검색 시 탐색 폭 (클수록 검색 정확도는 높아지지만 속도는 저하될 수 있음)
    :param num_threads: 검색에 사용할 스레드 수 (None이면 라이브러리 기본값)
    :param save_path: 인덱스 저장 경로
    :return: hnswlib Index
    """

    # 벡터 타입을 float32로 강제 (hnswlib 필수 사항)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    num, dim = vectors.shape

    # HNSW 인덱스 생성
    index = hnswlib.Index(space=space, dim=dim)

    # 인덱스 초기화 (최대 `num`개 벡터, 그래프 연결정도 `M`, 구축 탐색 폭 `ef_construction`)
    index.init_index(max_elements=num, M=M, ef_construction=ef_construction)

    # 내부 label index: 0 ~ num-1
    labels = np.arange(num, dtype=np.int32)
    index.add_items(vectors, labels)

    # 검색 시 탐색 폭 설정 (Recall <> 속도 trade-off)
    index.set_ef(ef)

    # 멀티스레드 검색 설정
    if num_threads is not None:
        index.set_num_threads(num_threads)

    if save_path is not None:
        index.save_index(str(save_path))

    return index


def search(
    query_vector: np.ndarray,
    index: hnswlib.Index,
    top_k: int = 5,
) -> Dict[int, float] | list[Dict[int, float]]:
    """
    쿼리 벡터(단일 또는 배치)에 대해 인덱스에서 근접 이웃 검색
    반환 형식:
      - 단일 벡터 입력: {내부인덱스: 점수}
      - 다중 벡터 입력: [{내부인덱스: 점수}, ...] (쿼리 순서 동일)

    :param query_vector: (dim,), (1, dim) 또는 (N, dim) 형태의 float32 배열
    :param index: hnswlib 인덱스
    :param top_k: 반환할 이웃 수
    """

    # -----------------------------------------------
    # 입력 벡터 형태 정규화
    # -----------------------------------------------
    is_single = query_vector.ndim == 1
    query = query_vector.reshape(1, -1) if is_single else query_vector
    if query.dtype != np.float32:
        query = query.astype(np.float32)

    # -----------------------------------------------
    # ANN 검색 수행 (배치 지원)
    # -----------------------------------------------
    labels, distances = index.knn_query(query, k=top_k)
    # hnswlib의 distance 해석
    # - "ip": 내적 점수 (클수록 유사)
    # - "cosine": 거리 (작을수록 유사) -> 유사도 = 1 - d
    # - "l2": L2 거리 (작을수록 유사) -> 유사도 = 1 / (1 + d)

    # -----------------------------------------------
    # 최종 반환할 결과 리스트 (쿼리마다 dict 저장)
    # -----------------------------------------------
    results: list[Dict[int, float]] = []
    for q in range(labels.shape[0]):
        # q번째 쿼리 벡터에 대한 검색 결과 저장용 dict
        result_q: Dict[int, float] = {}
        for rank, lbl in enumerate(labels[q]):
            # -1은 결과 없음(NULL)을 의미
            if lbl == -1:
                continue
                # hnswlib에서 -1이 나오는 경우 설명
                # - knn_query()는 항상 요청한 k개의 결과를 반환하려고 시도합니다.
                # - 하지만 실제로 인덱스에 들어있는 벡터 수가 부족하거나 특정 쿼리에 대해 유효한 이웃을 찾지 못했을 때,
                # - 결과 자리 채우기 용도로 -1을 반환합니다.

            # hnswlib가 반환한 거리(distance)
            score = float(distances[q][rank])

            # 거리 유사도 점수 변환
            if index.get_current_count() > 0:
                if index.space == "cosine":
                    score = 1 - score
                elif index.space == "l2":
                    score = 1 / (1 + score)
                # "ip"(inner product)는 변환 불필요 (내적 점수가 그대로 유사도)

            # 검색된 벡터의 내부 인덱스(lbl)에 유사도 점수 매핑
            result_q[int(lbl)] = score

        # q번째 쿼리의 결과 dict를 리스트에 추가 (예시: [ {idx1: score1, idx2: score2, ...}, {...}, ... ])
        results.append(result_q)

    return results[0] if is_single else results
