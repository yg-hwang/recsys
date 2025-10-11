import sys
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
from utils.retriever.vector_db import connect_milvus, build_filter_expr, search_milvus


# -----------------------------------------------
# 유저 및 상품 조회, API 호출 등을 실행할 함수
# -----------------------------------------------
dataset_dir = Path(root_dir).joinpath("data/dataset")
paths = DatasetPath(base_dir=dataset_dir, dataset_name="fashion")


@st.cache_resource
def get_database() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    유저, 상품 및 클릭스트림 데이터 로드 (DB라고 가정)
    """

    return (
        pd.read_parquet(paths.user_metadata_path),
        pd.read_parquet(paths.item_metadata_path),
        pd.read_parquet(paths.user_logs_path),
    )


df_user_metadata, df_item_metadata, df_user_logs = get_database()
df_item_metadata["name"] = df_item_metadata["name"].astype(str)


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


def predict_lightgcn(user_id: int, top_k: int = 100) -> dict:
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
    body = {"input_data": [{"user_id": user_id}], "top_k": top_k}

    # REST API 호출 -> JSON 응답 획득
    response = requests.post(url, json=body)

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
    body = [{"user_id": user_id, "inputs": inputs}]

    # REST API 호출 -> JSON 응답 획득
    response = requests.post(url, json=body)

    return response.json()


def show_candidates(
    item_ids: List[int],
    scores: List[float] = None,
    n_columns: int = 5,
    image_only: bool = False,
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
        if image_only:
            continue
        df_sub = df_sub.T.rename(columns={0: "상품 상세"})
        df_sub["상품 상세"] = df_sub["상품 상세"].astype(str)
        cols[idx].dataframe(df_sub)


# -----------------------------------------------
# 앱 화면 표시 부분 (UI)
# -----------------------------------------------
st.set_page_config(layout="wide")

if st.sidebar.button("새로고침"):
    st.cache_data.clear()
    st.cache_resource.clear()

MODEL = st.sidebar.selectbox(
    label="MODEL", options=["lightgcn", "transformer"], index=0
)
USER_ID = st.sidebar.text_input(
    label="USER ID", value="", placeholder="USER ID를 입력하세요."
)
TOP_K = st.sidebar.number_input(
    label="추천 상품 개수", min_value=10, max_value=100, value=50, step=5
)
IMAGE_ONLY = st.sidebar.toggle(label="썸네일만 보기", value=False)

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
            show_candidates(item_ids=item_ids, image_only=IMAGE_ONLY)

    if MODEL == "lightgcn":
        st.markdown("### 추천 상품")
        with st.container(border=True):
            response = predict_lightgcn(user_id=USER_ID, top_k=TOP_K)
            candidates = response["predictions"][0]["candidates"]
            item_ids = [int(item_id) for item_id in list(candidates.keys())]
            scores = list(candidates.values())
            show_candidates(item_ids=item_ids, scores=scores, image_only=IMAGE_ONLY)

    if MODEL == "transformer":
        st.markdown("### 상품 입력")
        with st.container(border=True):
            ITEM_IDS = st.text_input(
                label="상품 ID 시퀀스",
                value="",
                placeholder="상품 ID 입력 (예: 123 혹은 123, 234)",
            )

            if ITEM_IDS != "":
                item_ids = [int(item_id.strip()) for item_id in ITEM_IDS.split(",")]
            else:
                item_ids = []
                st.stop()

            cols = st.columns(len(item_ids))
            actions = []
            for i, col in enumerate(cols):
                ACTION = col.selectbox(
                    label=f"{i+1}) ID: **{item_ids[i]}**",
                    options=["click", "wishlist", "cart", "purchase"],
                    index=0,
                )
                actions.append(ACTION)

            if len(item_ids) > 5:
                st.warning(
                    f"현재 모델의 고정된 시퀀스 길이는 5입니다. 앞 쪽부터 다섯 개의 상품 시퀀스만 사용합니다. ({item_ids[:5]})"
                )

            with st.expander("상품 상세 보기"):
                show_candidates(item_ids=item_ids, image_only=IMAGE_ONLY)

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
            # 상위 key 기준 오름차순 정렬
            outputs = dict(sorted(outputs.items(), key=lambda x: x[0]))

            # 각 하위 dict을 value 기준 내림차순 정렬
            for key, sub_dict in outputs.items():
                sorted_sub = dict(
                    sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)
                )
                outputs[key] = sorted_sub

            query_vector = np.array(response["predictions"][0]["item_vector"])

            filter_expr = None
            with st.expander("예측값 및 필터링 보기"):
                col_1, col_2 = st.columns(2)

                col_1.markdown("#### 모델 예측값 원본")
                col_1.write(outputs)

                col_2.markdown("#### 필터링 조건")
                KEY = col_2.multiselect(
                    label="**1️⃣ 필터링 Feature**",
                    options=sorted(list(outputs.keys())),
                    placeholder="예: ['master_category', 'sub_category', 'article_type', 'gender']",
                )
                TOP_K_PER_FEATURE = col_2.number_input(
                    label="**2️⃣ Feature별 상위 label class 개수**",
                    min_value=2,
                    max_value=5,
                    step=1,
                    value=2,
                )
                PROBA_THRESHOLD = col_2.number_input(
                    label="**3️⃣ label class 확률 최솟값**",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=0.0,
                )
                filter_expr = build_filter_expr(
                    data=outputs,
                    key=KEY,
                    top_k_per_feature=TOP_K_PER_FEATURE,
                    proba_threshold=PROBA_THRESHOLD,
                )
                if filter_expr is None:
                    col_2.write(f"> **필터링 조건식이 없습니다.**")
                else:
                    col_2.write(f"> **{filter_expr}**")

            results = search_milvus(
                collection=collection,
                item_vector=query_vector,
                expr=filter_expr,
                limit=TOP_K,
            )

            candidates = {}
            for hit in results:
                candidates[int(hit.entity.get("item_id"))] = hit.score

            item_ids = list(candidates.keys())
            scores = list(candidates.values())
            show_candidates(item_ids=item_ids, scores=scores, image_only=IMAGE_ONLY)
