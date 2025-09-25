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
    """

    def __init__(self):
        super().__init__()
        self._all_classes = {}
        self._all_encoders = {}

    def fit(self, df: pd.DataFrame):

        for column in sorted(df.columns):
            le = LabelEncoder()
            values = df.loc[:, column].unique()
            le.fit(values)
            self._all_classes[column] = np.array(le.classes_.tolist(), dtype=object)
            self._all_encoders[column] = le
        logging.debug(">>> LabelEncoder created.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for column in sorted(df.columns):
            values = df.loc[:, column].to_numpy()
            encoded_values = self._all_encoders[column].transform(values)
            df.loc[:, column] = encoded_values
        logging.debug(">>> Encoding completed.")

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for column in sorted(df.columns):
            decoded_values = self._all_encoders[column].inverse_transform(
                df.loc[:, column].to_list()
            )
            df[column] = decoded_values

        return df

    @property
    def all_encoders(self):
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
        :param max_seq_len: 시퀀스 최대 길이
        :param user_id: 유저 식별 단위
        :param item_id: 상품 식별 단위
        :param order_by: 시퀀스 정렬 기준
        :param partition_by: (선택) 완성된 DataFrame을 파티셔닝하여 저장할 때 사용
        """
        self.max_seq_len = max_seq_len
        self.user_id = user_id
        self.item_id = item_id
        self.order_by = order_by
        self.partition_by = partition_by
        self.features = None
        self.targets = None

    def _check_columns(self, df: pd.DataFrame):
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
        max_seq_len보다 짧으면 0으로 패딩
        """
        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            return seq + [0] * (self.max_seq_len - seq_len)
        return seq[-self.max_seq_len :]

    def _create_mask(self, seq: List[Union[str, int]]) -> List[int]:
        """
        시퀀스 길이에 맞는 마스크 (0=실제값, 1=패딩)
        """
        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            return [0] * seq_len + [1] * (self.max_seq_len - seq_len)
        return [0] * self.max_seq_len

    def _sort_dataframe(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        if x is None:
            return None
        else:
            if seq_len < self.max_seq_len:
                return x[seq_len - 1]
            else:
                return x[-1]

    def get_seq_dataframe(
        self,
        data: pd.DataFrame,
        feature_sequences: List[str],
        output_targets: List[str] = None,
    ) -> pd.DataFrame:
        """
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

            # 타겟 생성
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
        self.feature_sequences: Dict[str, torch.Tensor] = {}
        self.masks: torch.Tensor
        self.targets: Union[Dict[str, torch.Tensor], None] = {}

        for feature in feature_sequences:
            if feature == "mask":
                x = np.array([np.array(x).astype(np.float32) for x in df[feature]])
                self.masks = torch.from_numpy(x).to(device)
            else:
                x = np.array([np.array(x).astype(np.int32) for x in df[feature]])
                self.feature_sequences[feature] = torch.from_numpy(x).to(device)

        if targets is not None:
            for target in targets:
                self.targets[target] = torch.from_numpy(
                    np.array([np.array(y).astype(np.float32) for y in df[target]])
                ).to(device)
        else:
            self.targets = None

    def __len__(self):
        feature = list(self.feature_sequences.keys())[0]
        return self.feature_sequences[feature].shape[0]

    def __getitem__(self, idx):
        feature_sequences = {
            feature_name: sequence[idx]
            for feature_name, sequence in self.feature_sequences.items()
        }
        if self.targets is not None:
            targets = {
                target_name: classes[idx]
                for target_name, classes in self.targets.items()
            }
            return feature_sequences, self.masks[idx], targets
        else:
            return feature_sequences, self.masks[idx]
