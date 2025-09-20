from typing import Sequence, Optional, Tuple
import numpy as np
import pandas as pd


# def generate_users(
#     num_users: int = 10000,
#     age_range: Tuple[int, int] = (18, 70),
#     genders: Sequence[str] = ("M", "F"),
#     gender_probs: Optional[Sequence[float]] = None,
#     seed: Optional[int] = None,
# ) -> pd.DataFrame:
#     """
#     유저 메타데이터( user_id, age, gender )를 랜덤으로 생성하여 DataFrame으로 반환합니다.
#
#     :param num_users: 생성할 유저 수
#     :param age_range: (최소나이, 최대나이) 범위. 양끝 포함.
#     :param genders: 성별 후보 값들. 기본 ("M", "F")
#     :param gender_probs: 성별 분포 확률. None이면 균등 분포
#     :param seed: 난수 시드 (재현성)
#     :return: columns = [user_id, age, gender]
#     """
#     if num_users <= 0:
#         return pd.DataFrame(columns=["user_id", "age", "gender"])
#
#     rng = np.random.default_rng(seed)
#
#     # 나이: 균등분포로 생성 (양끝 포함)
#     min_age, max_age = age_range
#     if min_age > max_age:
#         raise ValueError(
#             "age_range는 (min_age, max_age) 형태여야 하며 min_age <= max_age 이어야 합니다."
#         )
#     ages = rng.integers(min_age, max_age + 1, size=num_users)
#
#     # 성별: 주어진 확률로 샘플링 (없으면 균등)
#     if gender_probs is None:
#         gender_probs = [1.0 / len(genders)] * len(genders)
#     if abs(sum(gender_probs) - 1.0) > 1e-8:
#         raise ValueError("gender_probs의 합은 1이어야 합니다.")
#     genders_sampled = rng.choice(genders, size=num_users, p=gender_probs)
#
#     df = pd.DataFrame(
#         {
#             "user_id": np.arange(1, num_users + 1, dtype=int),
#             "age": ages.astype(int),
#             "gender": genders_sampled,
#         }
#     )
#     return df


def generate_users(
    num_users: int = 10000,
    age_range: Tuple[int, int] = (18, 70),
    genders: Sequence[str] = ("M", "F"),
    gender_probs: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    *,
    age_distribution: str = "triangular",  # "triangular" | "uniform"
    age_mode: Optional[float] = None,  # 나이 분포의 피크값(기본은 중앙값)
) -> pd.DataFrame:
    """
    유저 메타데이터( user_id, age, gender )를 생성하여 DataFrame으로 반환합니다.

    :param num_users: 생성할 유저 수
    :param age_range: (최소나이, 최대나이) 범위. 양끝 포함.
    :param genders: 성별 후보 값들. 기본 ("M", "F")
    :param gender_probs: 성별 분포 확률. None이면 균등 분포
    :param seed: 난수 시드 (재현성)
    :param age_distribution: 나이 생성 분포 ("triangular" | "uniform")
                            - "triangular": 중앙(또는 age_mode)에 가까울수록 더 많이 생성(기본)
                            - "uniform": 기존처럼 균등분포
    :param age_mode: 삼각분포의 피크 나이. None이면 age_range 중앙값 사용
    :return: columns = [user_id, age, gender]
    """
    if num_users <= 0:
        return pd.DataFrame(columns=["user_id", "age", "gender"])

    rng = np.random.default_rng(seed)

    # 나이 생성
    min_age, max_age = age_range
    if min_age > max_age:
        raise ValueError(
            "age_range는 (min_age, max_age) 형태여야 하며 min_age <= max_age 이어야 합니다."
        )

    if age_distribution not in ("triangular", "uniform"):
        raise ValueError(
            'age_distribution은 "triangular" 또는 "uniform" 이어야 합니다.'
        )

    if age_distribution == "uniform":
        # 기존 방식: 균등분포 (양끝 포함)
        ages = rng.integers(min_age, max_age + 1, size=num_users)
    else:
        # 삼각분포: 중앙(또는 age_mode)에 가까울수록 확률이 높음
        mode_val = (min_age + max_age) / 2.0 if age_mode is None else float(age_mode)
        if not (min_age <= mode_val <= max_age):
            raise ValueError("age_mode는 age_range 범위 내에 있어야 합니다.")

        # 경계 포함 정수화를 위해 좌우를 살짝 확장 후 반올림
        samples = rng.triangular(
            left=min_age - 0.499,
            mode=mode_val,
            right=max_age + 0.499,
            size=num_users,
        )
        ages = np.rint(samples).astype(int)
        ages = np.clip(ages, min_age, max_age)

    # 성별 생성
    if gender_probs is None:
        gender_probs = [1.0 / len(genders)] * len(genders)
    if abs(sum(gender_probs) - 1.0) > 1e-8:
        raise ValueError("gender_probs의 합은 1이어야 합니다.")
    genders_sampled = rng.choice(genders, size=num_users, p=gender_probs)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, num_users + 1, dtype=int),
            "age": ages.astype(int),
            "gender": genders_sampled,
        }
    )
    return df
