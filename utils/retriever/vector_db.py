import pymilvus
import numpy as np
from typing import List, Dict
from pymilvus import connections, Collection


# -----------------------------------------------
# Milvus 연결
# -----------------------------------------------
def connect_milvus(
    host: str = "localhost", port: str = "19530", collection_name: str = "fashion_items"
) -> Collection:
    """
    Milvus 서버에 연결하고 지정한 컬렉션을 메모리에 로드
    - Milvus는 서버형 벡터 데이터베이스이므로, Python 클라이언트(`pymilvus`)에서 쿼리를 실행하려면 먼저 서버(gRPC)로 연결해야 합니다.
    - 또한 벡터 검색 전에는 대상 컬렉션을 메모리에 로드(`load`)해야 ANN 인덱스가 활성화됩니다.

    :param host: Milvus 서버 호스트명 또는 IP (기본값: "localhost")
    :param port: Milvus gRPC 포트 (기본값: "19530")
    :param collection_name: 사용할 컬렉션명 (예: "fashion_items")
    :return: 로드가 완료된 `pymilvus.Collection` 객체

    Example:
        >>> collection = connect_milvus("localhost", "19530", "fashion_items")
        >>> # 이후 collection.search(...) 사용
    """
    connections.connect(alias="default", host=host, port=port)
    collection = Collection(collection_name)
    collection.load()

    return collection


# -----------------------------------------------
# Transformer 예측 결과(JSON) -> 필터 expr 생성
# -----------------------------------------------
def build_filter_expr(
    outputs: Dict[str, Dict[str, float]],
    top_k_per_feature: int = 1,
    key: List[str] = None,
) -> str:
    """
    Transformer 예측 결과를 Milvus Boolean filter expression으로 변환
    - 모델이 feature별로 예측한 label 확률 분포를 받아,
    - 각 feature에 대해 확률 상위 K개의 라벨만 사용하여 OR로 묶고,
    - feature 간에는 AND로 결합한 expr을 생성합니다.

    :param outputs: Transformer의 feature별 예측 결과
    :param top_k_per_feature: 각 feature에서 채택할 상위 라벨 개수
    :param key: 필터에 사용할 특정 feature 목록 (예: ['gender', 'article_type'])
    :return: Milvus `expr` 문자열 (예: 'gender == "Women" and article_type == "Tshirts"')

    Example:
        >>> outputs = {"gender": {"Women": 1.0}, "article_type": {"Tshirts": 0.99, "Shoes": 0.4}}
        >>> build_filter_expr(outputs, top_k_per_feature=1)
        'gender == "Women" and article_type == "Tshirts"'
    """

    if not outputs:
        raise ValueError(
            "`outputs`가 비어 있습니다. 최소 1개 이상의 feature가 필요합니다."
        )
    if top_k_per_feature < 1:
        raise ValueError("`top_k_per_feature`는 1 이상이어야 합니다.")

    expr_parts = []

    # key가 지정된 경우 해당 feature만 사용
    target_features = key if key is not None else list(outputs.keys())

    for feature in target_features:
        if feature not in outputs:
            continue

        probs = outputs[feature]
        if not probs:
            continue

        # 확률 상위 K개의 라벨 추출
        sorted_labels = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_labels = [label for label, _ in sorted_labels[:top_k_per_feature]]

        label_str = ", ".join([f'"{label}"' for label in top_labels])
        expr_parts.append(f"{feature} in [{label_str}]")

    expr = " and ".join(expr_parts) if expr_parts else ""

    return expr


# -----------------------------------------------
# item_vector -> Milvus 벡터 검색 수행
# -----------------------------------------------
def search_milvus(
    collection: Collection,
    item_vector: np.ndarray,
    expr: str,
    vector_field: str = "image_vector",
    limit: int = 10,
) -> List[pymilvus.client.abstract.Hit]:
    """
    예측된 item_vector를 Milvus에 질의하여, 선택한 vector 필드에서 ANN 검색 수행
    - 본 함수는 `metric_type="IP"` 를 사용헙니다.
    - 벡터가 L2 정규화되어 있다면, IP 검색은 cosine similarity와 동일한 순서를 보장합니다. (score는 1.0에 가까울수록 유사)

    :param collection: `connect_milvus()`로 로드된 Milvus 컬렉션 객체
    :param item_vector: 검색에 사용할 쿼리 벡터
    :param expr: Milvus Boolean filter 식 (예: 'gender == "Women" and article_type == "Tshirts"')
    :param vector_field: 검색할 벡터 필드명
    :param limit: 반환할 Top-K 개수

    :return: 검색 결과 리스트

    Example:
        >>> results = search_milvus(collection, item_vector, 'gender == "Women"', "image_vector", limit=5)
        >>> for hit in results:
        ...     print(hit.entity.get("item_id"), hit.score)
    """
    search_params = {"metric_type": "IP", "params": {"ef": 100}}

    results = collection.search(
        data=[item_vector.astype(np.float32)],
        anns_field=vector_field,
        param=search_params,
        limit=limit,
        expr=expr,
        output_fields=[
            "item_id",
            "name",
            "brand_name",
            "gender",
            "master_category",
            "article_type",
        ],
    )

    return results[0]
