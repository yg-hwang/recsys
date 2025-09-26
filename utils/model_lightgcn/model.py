import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List


# project_root 기준 경로 자동 계산
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.model_ann.ann import build_index, search


class Model:
    def __init__(self, model_dir: Union[str, Path]):

        self.model_dir = Path(model_dir).resolve()
        df_user_vectors = pd.read_parquet(
            model_dir.joinpath("lightgcn/user_vector.parquet")
        )
        df_item_vectors = pd.read_parquet(
            model_dir.joinpath("lightgcn/item_vector.parquet")
        )
        self.item_id_maps = dict(
            zip(df_item_vectors["idx"], df_item_vectors["item_id"])
        )
        self.user_vectors = dict(
            zip(
                df_user_vectors["user_id"],
                np.array(df_user_vectors["vector_normalized"].tolist()),
            )
        )
        item_vectors = np.array(df_item_vectors["vector_normalized"].tolist()).astype(
            np.float32
        )
        self.item_index = build_index(item_vectors)

    def preprocess(self, body: Dict[str, any]) -> Dict[str, any]:
        """
        모델 입력값 전처리

        ---------- 예시 입력 ----------
        {"user_id": 123}

        :param body: request body
        :return: body
        """

        user_id = body["user_id"]
        if user_id not in self.user_vectors:
            return body
        body["user_vector"] = np.array(self.user_vectors[user_id])

        return body

    def predict(self, input_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        예측 상품 ID 및 점수 반환

        :param input_data: request body
        :return: results
        """
        results = list()

        for d in input_data:
            data = self.preprocess(body=d)
            if data.get("user_vector") is None:
                results.append({"user_id": data["user_id"], "candidates": {}})
                continue
            result = search(data["user_vector"], self.item_index, top_k=50)

            item_indies = list(result.keys())[1:]

            candidates = {}
            for i in item_indies:
                item_id = self.item_id_maps[i]
                candidates[item_id] = result[i]

            results.append({"user_id": data["user_id"], "candidates": candidates})

        return results
