import torch
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from typing import Dict


class FeatureLabelEncoder(LabelEncoder):
    """
    범주형 컬럼을 정수형 컬럼으로 인코딩한다.

    - 각 컬럼마다 개별 LabelEncoder를 만들어 보관 및 재사용
    - `fit()`: 컬럼별 고유값으로 인코더 학습
    - `transform()`: 학습된 인코더로 각 컬럼을 정수로 치환 (in-place 할당)
    - `inverse_transform()`: 정수 -> 원래 라벨로 복원
    """

    def __init__(self):
        super().__init__()
        self._all_classes = {}
        self._all_encoders = {}

    def fit(self, df: pd.DataFrame):
        """
        전달된 DataFrame의 모든 컬럼에 대해 LabelEncoder를 학습합니다.
        """

        for column in sorted(df.columns):
            # 개별 feature LabelEncoder 생성
            le = LabelEncoder()

            # 해당 고유값 추출
            values = df.loc[:, column].unique()

            # 고유값 집합으로 인코더 학습
            le.fit(values)

            # classes_를 그대로 저장해두면 나중에 매핑을 외부로 내보낼 때도 유용
            self._all_classes[column] = np.array(le.classes_.tolist(), dtype=object)

            # 컬럼명 -> 인코더 매핑 저장
            self._all_encoders[column] = le
        logging.debug(">>> LabelEncoder created.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 인코더로 각 컬럼 값을 정수로 변환합니다.
        - in-place 할당을 수행하므로, df가 뷰(view)일 경우 pandas의 SettingWithCopyWarning이 발생할 수 있습니다.
          이 경우, 호출부에서 df = df.copy() 후 넘기는 것을 권장합니다.
        - 학습 시점에 없던 클래스 값(Unseen)이 등장하면 LabelEncoder는 에러를 발생시킵니다.
        """

        for column in sorted(df.columns):
            values = df.loc[:, column].to_numpy()
            encoded_values = self._all_encoders[column].transform(values)
            df.loc[:, column] = encoded_values
        logging.debug(">>> Encoding completed.")

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        정수로 인코딩된 각 컬럼을 원래 라벨로 복원합니다.
        - LabelEncoder.inverse_transform은 1D 배열을 기대하므로 컬럼별로 개별 호출합니다.
        """

        for column in sorted(df.columns):
            decoded_values = self._all_encoders[column].inverse_transform(
                df.loc[:, column].to_list()
            )
            df[column] = decoded_values

        return df

    @property
    def all_encoders(self):
        """
        컬럼별 LabelEncoder 딕셔너리 접근자
        - 예: encoder.all_encoders['category_id'].classes_ 로 클래스 목록 확인 가능
        """
        return self._all_encoders


class SequenceGenerator:
    def __init__(
        self,
        max_seq_len: int = 10,
        user_id: str = "user_id",
        item_id: str = "item_id",
        order_by: str = "timestamp",
        partition_by: str = None,
    ):
        """
        :param max_seq_len: 시퀀스 최대 길이 (초과하면 잘라내고, 부족하면 패딩)
        :param user_id: 유저 식별 컬럼명
        :param item_id: 상품 식별 컬럼명
        :param order_by: 시퀀스 정렬 기준 (보통 시간으로 함)
        :param partition_by: (선택) 완성된 DataFrame을 파티셔닝하여 저장할 때 사용
        """
        self.max_seq_len = max_seq_len
        self.user_id = user_id
        self.item_id = item_id
        self.order_by = order_by
        self.partition_by = partition_by

        # # 시퀀스 feature 컬럼명 저장
        self.features = None

        # target label 컬럼명 저장
        self.targets = None

    def _check_columns(self, df: pd.DataFrame):
        """
        필수 컬럼이 DataFrame에 있는지 확인
        """

        columns = df.columns
        if self.user_id not in columns:
            raise ValueError("`user_id` must be in columns.")
        if self.item_id not in columns:
            raise ValueError("`item_id` must be in columns.")
        if self.order_by not in columns:
            raise ValueError("`order_by` must be in columns.")
        if self.partition_by is not None and self.partition_by not in columns:
            raise ValueError(f"Not found '{self.partition_by}' column.")

    def _add_padding(self, seq: List[Union[str, int]]) -> List[Union[str, int]]:
        """
        max_seq_len보다 짧으면 뒤쪽에 0을 채워 길이를 맞춤 (post-padding)
        max_seq_len보다 길면 뒤쪽 recent max_seq_len 만큼만 남김 (truncation)
        """
        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            return seq + [0] * (self.max_seq_len - seq_len)
        return seq[-self.max_seq_len :]

    def _create_mask(self, seq: List[Union[str, int]]) -> List[int]:
        """
        시퀀스 길이에 맞는 mask (0=실제값, 1=패딩)
        - 모델 학습 시 padding 토큰을 무시하기 위해 필요
        """

        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            return [0] * seq_len + [1] * (self.max_seq_len - seq_len)
        return [0] * self.max_seq_len

    def _sort_dataframe(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        유저별 시퀀스 정렬 및 user_rn (유저 내 row 번호) 생성
        - user_rn은 시퀀스 내 몇 번째 이벤트인지 나타냄
        """

        data = data.sort_values([self.user_id, self.order_by])
        data["user_rn"] = data.groupby(self.user_id).cumcount() + 1

        if self.partition_by is not None:
            data_output = data[
                [self.user_id, self.item_id, "user_rn", self.partition_by]
            ].copy()
        else:
            data_output = data[[self.user_id, self.item_id, "user_rn"]].copy()

        return data, data_output

    def _extract_target(
        self, x: List[Union[str, int]], seq_len: int
    ) -> Union[int, str, None]:
        """
        시퀀스에서 target label 추출
        - 보통 마지막 아이템을 다음 예측의 타깃으로 사용
        """

        if x is None:
            return None
        else:
            if seq_len < self.max_seq_len:
                return x[seq_len - 1]  # 짧으면 마지막 원소

            else:
                return x[-1]  # 길면 잘린 뒤 마지막 원소

    def get_seq_dataframe(
        self,
        data: pd.DataFrame,
        feature_sequences: List[str],
        output_targets: List[str] = None,
    ) -> pd.DataFrame:
        """
        DataFrame을 시퀀스 데이터셋으로 변환

        :param data: Column-based DataFrame
        :param feature_sequences: Sequential Dataset으로 생성할 컬럼
        :param output_targets: Sequential Dataset으로 생성할 컬럼 중 Target Label 컬럼
        """

        self._check_columns(data)

        data, data_output = self._sort_dataframe(data)

        self.features = []
        self.targets = []

        for idx, col_name in enumerate(feature_sequences):
            self.features.append(col_name)

            # groupby-rolling 대신 custom 누적 시퀀스 생성
            df_seq = (
                data.groupby(self.user_id)[col_name]
                .apply(
                    lambda x: [
                        list(x.iloc[max(0, i - self.max_seq_len + 1) : i + 1])
                        for i in range(len(x))
                    ]
                )
                .explode()
                .reset_index(level=0, drop=True)
            )
            df_seq = df_seq.apply(list)

            if idx == 0:
                # 시퀀스 길이
                seq_len = df_seq.apply(len)
                mask = df_seq.apply(self._create_mask)
                df_seq = df_seq.apply(self._add_padding)

                data_output.loc[:, "seq_len"] = seq_len
                data_output.loc[:, "mask"] = mask
                data_output.loc[:, col_name] = df_seq

            else:
                df_seq = df_seq.apply(self._add_padding)
                data_output[col_name] = df_seq

            # target 생성
            if output_targets is not None and col_name in output_targets:
                target = f"y_{col_name}"
                self.targets.append(target)
                data_output.loc[:, target] = (
                    data.groupby(self.user_id)[col_name]
                    .shift(-1)
                    .reset_index(level=0, drop=True)
                )

        # 최종 컬럼 정리
        columns = (
            [self.user_id, self.item_id, "user_rn", "seq_len", "mask"]
            + self.features
            + self.targets
        )
        if self.partition_by is not None:
            columns.append(self.partition_by)

        data_output = (
            data_output[columns]
            .sort_values([self.user_id, "user_rn"])
            .reset_index(drop=True)
        )
        return data_output


class SequentialDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_sequences: List[str],
        targets: List[str] = None,
        device: str = "cpu",
    ):
        """
        Pandas DataFrame을 torch Dataset으로 변환

        :param df: 시퀀스 데이터셋 (SequenceGenerator 결과)
        :param feature_sequences: 입력 feature로 사용할 컬럼명 리스트
        :param targets: target label로 사용할 컬럼명 리스트
        :param device: 텐서를 저장할 장치 (CPU/GPU)
        """

        self.feature_sequences: Dict[str, torch.Tensor] = {}
        self.masks: torch.Tensor
        self.targets: Union[Dict[str, torch.Tensor], None] = {}

        # ---------- feature 시퀀스 준비 ---------- #
        for feature in feature_sequences:
            if feature == "mask":
                # mask는 float32 (0=실제, 1=패딩) -> 모델 attention mask에 직접 활용
                x = np.array([np.array(x).astype(np.float32) for x in df[feature]])
                self.masks = torch.from_numpy(x).to(device)
            else:
                # 나머지 feature는 정수형 시퀀스 (category_id 등)
                x = np.array([np.array(x).astype(np.int32) for x in df[feature]])
                self.feature_sequences[feature] = torch.from_numpy(x).to(device)

        # ---------- target label ---------- #
        if targets is not None:
            for target in targets:
                # float32로 변환 (분류 문제라면 이후 모델에서 softmax/CE loss 사용)
                self.targets[target] = torch.from_numpy(
                    np.array([np.array(y).astype(np.float32) for y in df[target]])
                ).to(device)
        else:
            self.targets = None

    def __len__(self):
        """
        데이터셋의 전체 샘플 개수 반환
        - feature_sequences 중 아무거나 하나 선택해서 길이 반환
        """

        feature = list(self.feature_sequences.keys())[0]
        return self.feature_sequences[feature].shape[0]

    def __getitem__(self, idx):
        """
        하나의 샘플(batch 단위 이전)을 반환
        - feature_sequences: {feature_name: 시퀀스 텐서}
        - mask: 해당 시퀀스의 패딩 마스크
        - targets: target label (있으면 반환, 없으면 None)
        """

        # 입력 feature 시퀀스 꺼내오기
        feature_sequences = {
            feature_name: sequence[idx]
            for feature_name, sequence in self.feature_sequences.items()
        }

        # target이 있으면 feature, mask, target 반환 (학습 및 검증용)
        if self.targets is not None:
            targets = {
                target_name: classes[idx]
                for target_name, classes in self.targets.items()
            }
            return feature_sequences, self.masks[idx], targets

        # target이 없으면 feature, mask만 반환 (추론용)
        else:
            return feature_sequences, self.masks[idx]


class SequenceVectorDataset(Dataset):
    def __init__(self, item_id: np.ndarray, seq_vector: np.ndarray, device: str):
        self.item_id = torch.from_numpy(item_id).to(device)
        self.seq_vector = torch.from_numpy(seq_vector).to(device)

    def __len__(self):
        return len(self.item_id)

    def __getitem__(self, idx):
        return self.item_id[idx], self.seq_vector[idx]
