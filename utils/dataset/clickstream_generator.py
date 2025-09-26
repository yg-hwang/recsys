import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class ClickstreamGenerator:
    def __init__(
        self,
        item_metadata_df: pd.DataFrame,
        similarity_keys: List[str],
        actions: List[str],
        action_weights: List[float],
        n_sessions_per_user: int,
        start_date: datetime,
        user_metadata_df: pd.DataFrame,  # 추가: 유저 메타데이터 전달
    ):
        self.item_metadata_df = item_metadata_df
        self.similarity_keys = similarity_keys
        self.actions = actions
        self.action_weights = action_weights
        self.n_sessions = n_sessions_per_user
        self.start_date = start_date
        self.item_metadata_dict = {
            row["item_id"]: row for _, row in item_metadata_df.iterrows()
        }
        # 추가: user_id -> {그 외 유저 속성들} 매핑
        self.user_attr_dict = {
            row["user_id"]: row.drop(labels=["user_id"]).to_dict()
            for _, row in user_metadata_df.iterrows()
        }
        self.similarity_cache = {}

    def get_similar_items(self, anchor_row):
        key = tuple(anchor_row[k] for k in self.similarity_keys)
        if key in self.similarity_cache:
            return self.similarity_cache[key]

        filtered_df = self.item_metadata_df
        for k in self.similarity_keys:
            filtered_df = filtered_df[filtered_df[k] == anchor_row[k]]

        similar_items = filtered_df["item_id"].tolist()
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
            anchor_row = self.item_metadata_df.sample(1).iloc[0]
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
    users_per_partition: int,
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
    item_metadata_df = pd.read_parquet(item_metadata_path)
    if "text_vector" in item_metadata_df.columns:
        item_metadata_df = item_metadata_df.drop(columns=["text_vector"])
    if "image_vector" in item_metadata_df.columns:
        item_metadata_df = item_metadata_df.drop(columns=["image_vector"])
    assert "item_id" in item_metadata_df.columns, "`item_id` 컬럼이 존재해야 합니다."

    # 유저 메타데이터 불러오기
    users_df = pd.read_parquet(user_metadata_path)
    assert (
        "user_id" in users_df.columns
    ), "user_metadata에는 `user_id` 컬럼이 존재해야 합니다."
    user_ids = users_df["user_id"].tolist()

    # 파티션별 클릭스트림 생성
    part = 0
    for start in range(0, len(user_ids), users_per_partition):
        generator = ClickstreamGenerator(
            item_metadata_df=item_metadata_df,
            user_metadata_df=users_df,
            similarity_keys=similarity_keys,
            actions=actions,
            action_weights=action_weights,
            n_sessions_per_user=n_sessions_per_user,
            start_date=start_date,
        )

        user_logs = []
        batch_user_ids = user_ids[start : start + users_per_partition]
        for u in batch_user_ids:
            user_logs.extend(generator.simulate_user_sessions(u))

        part_df = pd.DataFrame(user_logs)
        part_df.to_parquet(save_path.joinpath(f"part_{part:03d}"))
        part += 1
