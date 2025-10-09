import sys
import hnswlib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Tuple
from pymilvus import Collection


# -----------------------------------------------
# 프로젝트 경로 설정
# -----------------------------------------------
# setup_env를 import하기 위해 루트 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_env import setup_path

root_dir = setup_path()  # recsys 루트를 sys.path에 추가

from utils.dataset.config import DatasetPath
from utils.retriever.ann import build_index, search
from utils.retriever.vector_db import connect_milvus, build_filter_expr, search_milvus


# -----------------------------------------------
# 유저 및 상품 조회, API 호출 등을 실행할 함수
# -----------------------------------------------
dataset_dir = Path(root_dir).joinpath("data/dataset")
paths = DatasetPath(base_dir=dataset_dir, dataset_name="fashion")


@st.cache_resource
def get_metadata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    유저, 상품 및 클릭스트림 데이터 로드 (DB라고 가정)
    """

    return (
        pd.read_parquet(paths.user_metadata_path),
        pd.read_parquet(paths.item_metadata_path),
        pd.read_parquet(paths.user_logs_path),
    )


@st.cache_resource
def get_vector_index() -> Tuple[hnswlib.Index, dict]:
    """
    시퀀스 기반 상품 추천에 사용할 Vector 인덱스 로드 (임시 Vector DB라고 가정)
    """
    df_image_vectors = pd.read_parquet(paths.image_vectors_path)
    image_vectors = np.array(df_image_vectors["image_vector"].tolist())
    image_vectors = np.array(image_vectors).astype(np.float32)
    item_index = build_index(image_vectors)
    item_id_maps = dict(
        zip(df_image_vectors["item_id"].index, df_image_vectors["item_id"])
    )

    return item_index, item_id_maps


df_user_metadata, df_item_metadata, df_user_logs = get_metadata()
df_item_metadata["name"] = df_item_metadata["name"].astype(str)
item_index, item_id_maps = get_vector_index()


@st.cache_resource
def get_milvus_connection() -> Collection:
    """
    시퀀스 기반 상품 추천에 사용할 Milvus Vector Database 연결 객체
    """
    return connect_milvus()


collection = connect_milvus()


@st.cache_resource
def get_user_info(user_id: int) -> pd.DataFrame:
    """
    유저 정보 조회
    """
    return df_user_metadata[df_user_metadata["user_id"] == user_id].reset_index(
        drop=True
    )


@st.cache_resource
def get_item_info(item_ids: List[int]) -> pd.DataFrame:
    """
    상품 정보 조회 조회
    """
    return df_item_metadata[df_item_metadata["item_id"].isin(item_ids)].reset_index(
        drop=True
    )


@st.cache_resource
def get_user_logs(user_id: int) -> pd.DataFrame:
    """
    유저 클릭 및 구매 이력 조회
    """
    return (
        df_user_logs[df_user_logs["user_id"] == user_id]
        .drop(columns=["user_id", "age"])
        .sort_values("timestamp", ascending=False, ignore_index=True)
    )


@st.cache_resource
def search_vectors(query_vector: np.ndarray, top_k: int = 30) -> dict:
    result = search(query_vector, item_index, top_k=top_k)

    # 추천 아이템 Index (원본 ID가 아닌 별도의 인덱스)
    item_indies = list(result.keys())

    # 추천 아이템 ID 재매핑
    candidates = {}
    for i in item_indies:
        # 원본 ID로 매핑
        item_id = item_id_maps[i]
        candidates[int(item_id)] = result[i]

    return candidates


def predict_lightgcn(user_id: int) -> dict:
    """
    LightGCN 기반 추천 상품 예측

    예상 응답 형식
    ```
    {
        "predictions": [
            {
                "user_id": 123,
                "candidates": {"1064": 0.4437575340270996, , ..., "1268": 0.494867205619812},
            }
        ]
    }
    ```
    """
    url = "http://localhost:3000/predict_lightgcn"

    # 입력: user_id (추천 대상 유저)
    payload = [{"user_id": user_id}]

    # REST API 호출 -> JSON 응답 획득
    response = requests.post(url, json=payload)

    return response.json()


def predict_transformer(user_id: int, inputs: dict) -> dict:
    """
    Transformer 기반 추천 상품 예측

    예상 응답 형식
    ```
    {
        "predictions": [
            {
                "user_id": 123,
                "seq_vector": [-0.1724550575017929, ..., 0.2086903303861618],
                "item_vector": [0.026774995028972626, ..., -0.007867199368774891],
                "outputs": {
                    "age_group": {
                        "Adults-Men": 0.9999995231628418,
                        "Adults-Women": 0.9999988675117493,
                    },
                    "article_type": {
                        "Flip Flops": 0.9999811053276062,
                        "Shirts": 0.999975860118866,
                    },
                    "base_color": {
                        "Blue": 0.9999997615814209,
                        "Gold": 0.999974250793457,
                        "Grey": 0.9999953508377075,
                        "Pink": 0.9999836683273315,
                        "White": 0.999998927116394,
                    },
                    "brand_name": {
                        "Basics": 0.9999939203262329,
                        "Wrangler": 0.9999992847442627,
                        "iPanema": 0.9999687671661377,
                    },
                    "fit": {
                        "<UNK>": 0.9999997615814209,
                        "Regular Fit": 0.9999995231628418,
                    },
                    "gender": {"Men": 1.0, "Women": 0.9999988675117493},
                    "master_category": {
                        "Apparel": 0.9999996423721313,
                        "Footwear": 0.9999868273735046,
                    },
                    "occasion": {
                        "<UNK>": 0.9999982118606567,
                        "Casual": 0.9999992847442627,
                    },
                    "season": {
                        "Summer": 0.9999997615814209,
                        "Winter": 0.9999963641166687,
                    },
                    "sub_category": {
                        "Flip Flops": 0.9999814033508301,
                        "Topwear": 0.9999996423721313,
                    },
                    "usage": {"Casual": 0.9999998211860657},
                    "year": {"2012": 0.9999987483024597},
                },
            }
        ]
    }
    ```
    """
    url = "http://localhost:3000/predict_transformer"

    # 입력: user_id & 상품 시퀀스
    payload = [{"user_id": user_id, "inputs": inputs}]

    # REST API 호출 -> JSON 응답 획득
    response = requests.post(url, json=payload)

    return response.json()


def show_candidates(
    item_ids: List[int], scores: List[float] = None, n_columns: int = 5
) -> None:
    """
    상품 화면 표시
    """

    df_candidates = get_item_info(item_ids=item_ids)

    if df_candidates.shape[0] == 0:
        st.error("상품이 존재하지 않습니다. `item_id`를 확인하세요.")

    df_candidates["item_id"] = df_candidates["item_id"].astype(int)

    if scores is not None:
        data = {int(item_id): score for item_id, score in zip(item_ids, scores)}
        df_candidates["score"] = df_candidates["item_id"].map(data)
        df_candidates = df_candidates.sort_values(
            "score", ascending=False, ignore_index=True
        )

    cols = st.columns(n_columns)
    for i, item_id in enumerate(item_ids):
        img_path = paths.base_path.joinpath(f"images/{item_id}.jpg")
        df_sub = df_candidates[df_candidates["item_id"] == item_id].reset_index(
            drop=True
        )
        idx = i % n_columns
        cols[idx].image(img_path, width="stretch")
        if scores is not None:
            cols[idx].caption(f"{i+1}) **Score: {round(df_sub['score'].item(), 4)}**")
        df_sub = df_sub.T.rename(columns={0: "상품 상세"})
        df_sub["상품 상세"] = df_sub["상품 상세"].astype(str)
        cols[idx].dataframe(df_sub)


# -----------------------------------------------
# 앱 화면 표시 부분 (UI)
# -----------------------------------------------
st.set_page_config(layout="wide")

if st.button("새로고침"):
    st.cache_data.clear()
    st.cache_resource.clear()


MODEL = st.sidebar.selectbox(
    label="MODEL", options=["lightgcn", "transformer"], index=0
)


USER_ID = st.sidebar.text_input(
    label="USER ID", value="", placeholder="USER ID를 입력하세요."
)

if USER_ID != "":
    USER_ID = int(USER_ID)
    st.markdown("### 사용자 정보")
    with st.container(border=True):
        user_info = get_user_info(USER_ID)
        cols = st.columns(3)
        cols[0].metric("USER ID", USER_ID)
        cols[1].metric("AGE", user_info["age"].item())
        cols[2].metric("GENDER", user_info["gender"].item())

        with st.expander("탐색 및 구매 이력"):
            df_user_history = get_user_logs(user_id=USER_ID)
            item_ids = df_user_history["item_id"].astype(int).tolist()
            st.write(df_user_history)
            show_candidates(item_ids=item_ids)

    if MODEL == "lightgcn":
        st.markdown("### 추천 상품")
        with st.container(border=True):
            response = predict_lightgcn(user_id=USER_ID)
            candidates = response["predictions"][0]["candidates"]
            item_ids = [int(item_id) for item_id in list(candidates.keys())]
            scores = list(candidates.values())
            show_candidates(item_ids=item_ids, scores=scores)

    if MODEL == "transformer":
        st.markdown("### 상품 입력")
        with st.container(border=True):
            col_1, col_2 = st.columns(2)
            ITEM_IDS = col_1.text_input(
                label="상품 ID",
                value="",
                placeholder="두 개 이상 입력 시 쉼표 구분 (예: 123, 234)",
            )
            ACTIONS = col_2.text_input(
                label="(선택) 행동 시퀀스",
                value="",
                placeholder="두 개 이상 입력 시 쉼표 구분 (예: click, wishlist)",
            )
            if ITEM_IDS != "":
                item_ids = [int(item_id.strip()) for item_id in ITEM_IDS.split(",")]
                with st.expander("상품 상세 보기"):
                    show_candidates(item_ids=item_ids)
            else:
                item_ids = []
                st.stop()

            if ACTIONS:
                actions = [action.strip() for action in ACTIONS.split(",")]
                if len(actions) > len(item_ids):
                    col_2.error(
                        f"행동 시퀀스가 상품의 길이와 맞지 않습니다. (상품 시퀀스 길이: {len(item_ids)})"
                    )
                    st.stop()

                for a in actions:
                    if a not in ["click", "wishlist", "cart", "purchase"]:
                        col_2.error(
                            "올바른 값을 입력하세요. (입력 가능 값: **`click`**, **`wishlist`**, **`cart`**, **`purchase`**)"
                        )
                        st.stop()
            else:
                actions = []

        st.markdown("### 추천 상품")
        with st.container(border=True):

            df_item_info = get_item_info(item_ids=item_ids)
            input_columns = [
                "age_group",
                "article_type",
                "base_color",
                "brand_name",
                "fit",
                "gender",
                "master_category",
                "occasion",
                "season",
                "sub_category",
                "usage",
                "year",
            ]
            inputs = df_item_info[input_columns].to_dict(orient="list")
            if actions:
                inputs["action"] = actions
            response = predict_transformer(user_id=USER_ID, inputs=inputs)

            outputs = response["predictions"][0]["outputs"]
            query_vector = np.array(response["predictions"][0]["item_vector"])

            expr = build_filter_expr(
                outputs=outputs,
                key=["master_category", "sub_category", "article_type", "gender"],
                top_k_per_feature=3,
            )

            with st.expander("예측값 및 필터링 보기"):
                col_1, col_2 = st.columns(2)
                col_1.write(outputs)
                col_2.write(expr)

            results = search_milvus(
                collection=collection, item_vector=query_vector, expr=expr, limit=100
            )

            candidates = {}
            for hit in results:
                candidates[int(hit.entity.get("item_id"))] = hit.score

            item_ids = list(candidates.keys())
            scores = list(candidates.values())
            show_candidates(item_ids=item_ids, scores=scores)
