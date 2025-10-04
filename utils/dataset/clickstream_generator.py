import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta


class ClickstreamGenerator:
    """
    가상 클릭스트림(유저 세션 로그)을 생성하는 클래스
    - 유저가 아이템을 탐색하고 행동(action)을 수행하는 로그를 시뮬레이션
    """

    def __init__(
        self,
        df_item_metadata: pd.DataFrame,
        df_user_metadata: pd.DataFrame,
        similarity_keys: List[str],
        actions: List[str],
        action_weights: List[float],
        n_sessions_per_user: int,
        start_date: datetime,
    ):
        self.df_item_metadata = df_item_metadata

        # 유사 아이템을 판단할 기준 속성 (예: color, style)
        self.similarity_keys = similarity_keys

        # 가능한 행동 (예: view, click, purchase)
        self.actions = actions

        # 행동별 확률 분포
        self.action_weights = action_weights

        # 유저당 세션 수
        self.n_sessions = n_sessions_per_user
        self.start_date = start_date

        # item_id -> 아이템 속성 row 매핑
        self.item_metadata_dict = {
            row["item_id"]: row for _, row in df_item_metadata.iterrows()
        }

        # `user_id` -> 유저 속성 dict 매핑 (`user_id` 제외 나머지 속성만 저장)
        self.user_attr_dict = {
            row["user_id"]: row.drop(labels=["user_id"]).to_dict()
            for _, row in df_user_metadata.iterrows()
        }

        # 동일 `anchor_row` 기준 유사 아이템 캐싱 (성능 최적화)
        self.similarity_cache = {}

    def get_similar_items(self, anchor_row):
        """
        anchor 아이템과 유사한 아이템 목록 조회
        - `similarity_keys` 컬럼 값이 동일한 아이템들을 필터링
        - 캐싱을 활용해 반복 계산 방지
        """
        key = tuple(anchor_row[k] for k in self.similarity_keys)
        if key in self.similarity_cache:
            return self.similarity_cache[key]

        df_filtered = self.df_item_metadata
        for k in self.similarity_keys:
            df_filtered = df_filtered[df_filtered[k] == anchor_row[k]]

        similar_items = df_filtered["item_id"].tolist()
        self.similarity_cache[key] = similar_items
        return similar_items

    def simulate_user_sessions(self, user_id: str) -> List[Dict]:
        """
        특정 유저의 세션 로그 생성
        - anchor 아이템을 랜덤 선택 -> 유사 아이템 일부 탐색
        - 각 아이템에 대해 행동(action), timestamp 부여
        - user metadata, item metadata 병합
        """
        session_rows = []
        for _ in range(self.n_sessions):

            # -----------------------------------------------
            # 세션 시작 시각 (start_date 기준 랜덤 offset)
            # -----------------------------------------------
            base_time = self.start_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )

            # -----------------------------------------------
            # anchor 아이템 랜덤 선택
            # -----------------------------------------------
            anchor_row = self.df_item_metadata.sample(1).iloc[0]

            # -----------------------------------------------
            # anchor 아이템과 유사한 아이템 추출
            # -----------------------------------------------
            similar_items = self.get_similar_items(anchor_row)
            if not similar_items:
                continue

            # -----------------------------------------------
            # 유저가 실제로 본 아이템 리스트 (3~6개 랜덤 선택)
            # -----------------------------------------------
            viewed_items = random.sample(
                similar_items, min(len(similar_items), random.randint(3, 6))
            )

            # -----------------------------------------------
            # 해당 user_id의 속성 조회 (없으면 빈 dict)
            # -----------------------------------------------
            user_attrs = self.user_attr_dict.get(user_id, {})

            # -----------------------------------------------
            # `viewed_items` 각각에 대해 로그 생성
            # -----------------------------------------------
            for i, item_id in enumerate(viewed_items):
                timestamp = base_time + timedelta(minutes=i * random.randint(1, 4))
                action = random.choices(self.actions, weights=self.action_weights)[0]
                item_row = self.item_metadata_dict[item_id]
                row = {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": action,
                    # 유저 메타데이터 컬럼 병합
                    **user_attrs,
                    # 아이템 속성 병합
                    **item_row.drop(labels=["item_id"]).to_dict(),
                }
                session_rows.append(row)
        return session_rows


def generate_clickstream(
    item_metadata_path: Union[str, Path],
    user_metadata_path: Union[str, Path],
    save_path: Path,
    users_per_chunk: int,
    n_sessions_per_user: int,
    actions: List[str],
    action_weights: List[float],
    similarity_keys: List[str],
    start_date: datetime,
    seed: Optional[int] = None,
):
    """
    클릭스트림 로그 데이터셋 생성 함수
    - item_metadata, user_metadata parquet 파일을 불러와 시뮬레이션 실행
    - `users_per_chunk` 단위로 나눠서 parquet 파일로 저장
    """

    if seed is not None:
        random.seed(seed)

    save_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------
    # 아이템 메타데이터 로드
    # -----------------------------------------------
    df_item_metadata = pd.read_parquet(item_metadata_path)

    # 벡터는 로그에 필요 없으므로 제거
    if "text_vector" in df_item_metadata.columns:
        df_item_metadata = df_item_metadata.drop(columns=["text_vector"])
    if "image_vector" in df_item_metadata.columns:
        df_item_metadata = df_item_metadata.drop(columns=["image_vector"])
    assert "item_id" in df_item_metadata.columns, "`item_id` 컬럼이 존재해야 합니다."

    # -----------------------------------------------
    # 유저 메타데이터 로드
    # -----------------------------------------------
    df_users = pd.read_parquet(user_metadata_path)
    assert (
        "user_id" in df_users.columns
    ), "user_metadata에는 `user_id` 컬럼이 존재해야 합니다."
    user_ids = df_users["user_id"].tolist()

    # -----------------------------------------------
    # 유저를 chunk 단위로 로그 생성
    # -----------------------------------------------
    chunk = 0
    for start in range(0, len(user_ids), users_per_chunk):
        generator = ClickstreamGenerator(
            df_item_metadata=df_item_metadata,
            df_user_metadata=df_users,
            similarity_keys=similarity_keys,
            actions=actions,
            action_weights=action_weights,
            n_sessions_per_user=n_sessions_per_user,
            start_date=start_date,
        )

        user_logs = []
        batch_user_ids = user_ids[start : start + users_per_chunk]
        for u in batch_user_ids:
            user_logs.extend(generator.simulate_user_sessions(u))

        df_chunk = pd.DataFrame(user_logs)
        df_chunk.to_parquet(save_path.joinpath(f"chunk_{chunk:03d}.parquet"))
        chunk += 1
