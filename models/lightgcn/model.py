import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List


# -----------------------------------------------
# 프로젝트 경로 설정
# -----------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ANN 빌드 및 검색 함수 불러오기
from utils.retriever.ann import build_index, search


class Model:
    def __init__(self, model_dir: Union[str, Path]) -> None:
        """
        ANN 기반 추천 모델
        - 학습된 LightGCN 결과를 불러와 ANN 인덱스를 빌드하고 추천 결과 제공

        :param model_dir: 학습된 모델 임베딩 벡터(weights)가 저장된 디렉토리
        """

        # -----------------------------------------------
        # 모델 경로 및 데이터 로드
        # -----------------------------------------------
        self.model_dir = Path(model_dir).resolve()
        df_user_vectors = pd.read_parquet(model_dir.joinpath("user_vector.parquet"))
        df_item_vectors = pd.read_parquet(model_dir.joinpath("item_vector.parquet"))

        # -----------------------------------------------
        # 매핑 딕셔너리 생성
        # -----------------------------------------------
        # ANN 인덱스의 내부 idx와 실제 item_id 매핑
        self.item_id_maps = dict(
            zip(df_item_vectors["idx"], df_item_vectors["item_id"])
        )

        # user_id와 유저 벡터 매핑
        self.user_vectors = dict(
            zip(
                df_user_vectors["user_id"],
                np.array(df_user_vectors["vector_normalized"].tolist()),
            )
        )

        # -----------------------------------------------
        # ANN 인덱스 빌드
        # -----------------------------------------------
        # item 벡터 전체를 numpy float32 배열로 변환
        item_vectors = np.array(df_item_vectors["vector_normalized"].tolist()).astype(
            np.float32
        )

        # ANN 인덱스 생성 (HNSW)
        self.item_index = build_index(item_vectors)

    def preprocess(self, body: Dict[str, any]) -> Dict[str, any]:
        """
        모델 입력값 전처리
        - 유저 ID에 유저 벡터를 추가

        ---------- 예시 입력 ----------
        {"user_id": 123}

        :param body: request body
        :return: body (user_vector 추가된 상태)
        """

        user_id = body["user_id"]

        # 해당 유저가 벡터 사전에 없는 경우 (cold-start 상황)
        if user_id not in self.user_vectors:
            return body

        # user_vector를 body에 추가
        body["user_vector"] = np.array(self.user_vectors[user_id])

        return body

    def predict(
        self, input_data: List[Dict[str, any]], top_k: int = 100
    ) -> List[Dict[str, any]]:
        """
        추천 아이템 ID 및 점수 반환

        :param input_data: request body 리스트 (각 원소에 user_id 포함)
        :param top_k: 상품 후보군 개수
        :return: results
                 예시:
                 [
                     {
                         "user_id": 123,
                         "candidates": {item_id1: score1, item_id2: score2, ...}
                     },
                     ...
                 ]
        """
        results = list()

        for d in input_data:
            # -----------------------------------------------
            # 전처리: user_id -> user_vector 조회
            # -----------------------------------------------
            data = self.preprocess(body=d)

            # 유저 벡터가 없으면 빈 결과 반환
            if data.get("user_vector") is None:
                results.append({"user_id": data["user_id"], "candidates": {}})
                continue

            # -----------------------------------------------
            # ANN 검색: 유저 벡터 기준 Top-K 아이템 후보 검색
            # -----------------------------------------------
            result = search(data["user_vector"], self.item_index, top_k=top_k)

            # 추천 상품 후보군 인덱스
            item_indies = list(result.keys())

            # -----------------------------------------------
            # 내부 idx -> 실제 item_id 변환
            # -----------------------------------------------
            candidates = {}
            for i in item_indies:
                item_id = self.item_id_maps[i]
                candidates[item_id] = result[i]
            candidates = dict(
                sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            )

            # -----------------------------------------------
            # 결과 저장 (입력 유저가 여러 개일 수 있음)
            # -----------------------------------------------
            results.append({"user_id": data["user_id"], "candidates": candidates})

        return results
