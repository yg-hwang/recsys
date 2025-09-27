import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class ClickstreamGenerator:
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
        self.similarity_keys = similarity_keys
        self.actions = actions
        self.action_weights = action_weights
        self.n_sessions = n_sessions_per_user
        self.start_date = start_date
        self.item_metadata_dict = {
            row["item_id"]: row for _, row in df_item_metadata.iterrows()
        }
        # 추가: user_id -> {그 외 유저 속성들} 매핑
        self.user_attr_dict = {
            row["user_id"]: row.drop(labels=["user_id"]).to_dict()
            for _, row in df_user_metadata.iterrows()
        }
        self.similarity_cache = {}

    def get_similar_items(self, anchor_row):
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
        session_rows = []
        for _ in range(self.n_sessions):
            base_time = self.start_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            anchor_row = self.df_item_metadata.sample(1).iloc[0]
            similar_items = self.get_similar_items(anchor_row)
            if not similar_items:
                continue

            viewed_items = random.sample(
                similar_items, min(len(similar_items), random.randint(3, 6))
            )

            # 추가: 해당 user_id의 전체 속성 확보(없으면 빈 dict)
            user_attrs = self.user_attr_dict.get(user_id, {})

            for i, item_id in enumerate(viewed_items):
                timestamp = base_time + timedelta(minutes=i * random.randint(1, 4))
                action = random.choices(self.actions, weights=self.action_weights)[0]
                item_row = self.item_metadata_dict[item_id]
                row = {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": action,
                    **user_attrs,  # 추가: 유저 메타데이터 컬럼 병합
                    **item_row.drop(labels=["item_id"]).to_dict(),
                }
                session_rows.append(row)
        return session_rows


def generate_clickstream(
    item_metadata_path: str,
    user_metadata_path: str,
    save_path: Path,
    users_per_chunk: int,
    n_sessions_per_user: int,
    actions: List[str],
    action_weights: List[float],
    similarity_keys: List[str],
    start_date: datetime,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)

    save_path.mkdir(parents=True, exist_ok=True)

    # 상품 메타데이터 불러오기
    df_item_metadata = pd.read_parquet(item_metadata_path)
    if "text_vector" in df_item_metadata.columns:
        df_item_metadata = df_item_metadata.drop(columns=["text_vector"])
    if "image_vector" in df_item_metadata.columns:
        df_item_metadata = df_item_metadata.drop(columns=["image_vector"])
    assert "item_id" in df_item_metadata.columns, "`item_id` 컬럼이 존재해야 합니다."

    # 유저 메타데이터 불러오기
    df_users = pd.read_parquet(user_metadata_path)
    assert (
        "user_id" in df_users.columns
    ), "user_metadata에는 `user_id` 컬럼이 존재해야 합니다."
    user_ids = df_users["user_id"].tolist()

    # chunk별 클릭스트림 생성
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
        df_chunk.to_parquet(save_path.joinpath(f"chunk_{chunk:03d}"))
        chunk += 1
