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
    범주형 컬럼을 연속형 정수로 인코딩하기 위한 기능

    - 각 컬럼마다 개별 LabelEncoder를 만들어 보관 및 재사용 (특수 토큰 '-1', '<UNK>'를 강제로 포함)
    - `fit()`: 컬럼별 고유값으로 인코더 학습
    - `transform()`: 학습된 인코더로 각 컬럼을 정수로 치환 (in-place 할당)
    - `inverse_transform()`: 정수 -> 원래 label로 복원
    """

    def __init__(self):
        super().__init__()
        self._all_classes = {}  # 컬럼별 고유 label class 저장
        self._all_encoders = {}  # 컬럼별 LabelEncoder 객체 저장
        self.special_tokens = ["-1", "<UNK>"]

    def fit(self, df: pd.DataFrame):
        """
        전달된 DataFrame의 모든 컬럼에 대해 LabelEncoder를 생성
        - '-1', '<UNK>' 토큰을 반드시 포함
        """

        for column in sorted(df.columns):
            # 개별 feature LabelEncoder 생성
            le = LabelEncoder()

            # 고유값 추출 후, 특수 토큰을 강제로 포함
            values = df.loc[:, column].astype(str).unique().tolist()
            values = list(set(values) | set(self.special_tokens))

            # 고유값 집합으로 인코더 학습
            le.fit(values)

            # classes_를 그대로 저장해두면 나중에 매핑을 외부로 내보낼 때도 유용
            self._all_classes[column] = np.array(le.classes_.tolist(), dtype=object)

            # 컬럼명 -> 인코더 매핑 저장
            self._all_encoders[column] = le
        logging.debug(">>> LabelEncoder created.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 인코더로 각 컬럼 값을 정수로 변환
        """

        for column in sorted(df.columns):
            values = df.loc[:, column].astype(str).to_numpy()
            encoded_values = self._all_encoders[column].transform(values)
            df.loc[:, column] = encoded_values  # 정수 인코딩된 값으로 덮어쓰기
        logging.debug(">>> Encoding completed.")

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        정수로 인코딩된 각 컬럼을 원래 label로 복원
        - LabelEncoder.inverse_transform은 1D 배열을 기대하므로 컬럼별로 개별 호출
        """

        for column in sorted(df.columns):
            decoded_values = self._all_encoders[column].inverse_transform(
                df.loc[:, column].to_list()
            )
            df[column] = decoded_values  # 원본 값으로 되돌림

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
        시퀀스 데이터셋 생성 클래스
        - 개별 유저의 행동 로그를 시퀀스 형태로 변환
        - Transformer 등 Sequential 모델 학습에 필요한 입력 형식으로 준비

        :param max_seq_len: 시퀀스 최대 길이 (초과하면 잘라내고, 부족하면 패딩)
        :param user_id: 유저 식별 컬럼명
        :param item_id: 아이템 식별 컬럼명
        :param order_by: 시퀀스 정렬 기준 (보통 시간으로 함)
        :param partition_by: (선택) 완성된 DataFrame을 파티셔닝하여 저장할 때 사용
        """
        self.max_seq_len = max_seq_len
        self.user_id = user_id
        self.item_id = item_id
        self.order_by = order_by
        self.partition_by = partition_by

        # 시퀀스 feature 컬럼명 저장
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
        유저별 시퀀스 정렬 및 유저 내 row 번호(`user_rn`) 생성
        - `user_rn`은 시퀀스 내 몇 번째 이벤트인지 나타냄
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
        - 보통 마지막 아이템을 다음 예측의 target으로 사용
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
        - 유저별 행동 데이터를 rolling window처럼 시퀀스로 변환
        - padding/mask/target까지 포함된 구조로 반환

        :param data: 입력 DataFrame (user_id, item_id, timestamp 포함)
        :param feature_sequences: 시퀀스로 만들 feature 컬럼 리스트
        :param output_targets: Target Label로 사용할 feature 컬럼 리스트
        """

        # (1) 필수 컬럼 체크
        self._check_columns(data)

        # (2) 정렬 + user_rn 생성
        data, data_output = self._sort_dataframe(data)

        self.features = []
        self.targets = []
        total_skipped = 0  # skip된 시퀀스 수 카운트

        # (3) 유저별 시퀀스 생성 함수 (마지막 짧은 건 skip)
        def make_sequences(x: pd.Series) -> list:
            nonlocal total_skipped
            seqs = []
            n = len(x)
            for i in range(n):
                seq = list(x.iloc[max(0, i - self.max_seq_len + 1) : i + 1])
                if i == n - 1 and len(seq) < self.max_seq_len:
                    # 마지막인데 길이가 부족하면 skip
                    total_skipped += 1
                    continue
                seqs.append(seq)

            return seqs

        for idx, col_name in enumerate(feature_sequences):
            self.features.append(col_name)

            # 입력 시퀀스 생성
            df_seq = (
                data.groupby(self.user_id)[col_name]
                .apply(make_sequences)
                .explode()
                .reset_index(level=0, drop=True)
            ).apply(list)

            if idx == 0:
                # 첫 feature 처리 시 seq_len, mask 추가
                seq_len = df_seq.apply(len)
                mask = df_seq.apply(self._create_mask)
                df_seq = df_seq.apply(self._add_padding)

                data_output = data_output.loc[df_seq.index].copy()
                data_output["seq_len"] = seq_len
                data_output["mask"] = mask
                data_output[col_name] = df_seq
            else:
                df_seq = df_seq.loc[data_output.index]
                data_output[col_name] = df_seq.apply(self._add_padding)

            # (4) target 시퀀스 생성
            if output_targets is not None and col_name in output_targets:
                shifted_col = f"__shift__{col_name}"
                data[shifted_col] = data.groupby(self.user_id)[col_name].shift(-1)

                df_tgt = (
                    data.groupby(self.user_id)[shifted_col]
                    .apply(make_sequences)
                    .explode()
                    .reset_index(level=0, drop=True)
                ).apply(list)

                df_tgt = df_tgt.apply(
                    lambda s: [v if pd.notnull(v) else 0 for v in s]
                ).apply(self._add_padding)

                # 입력과 동일한 index만 유지 (skip 동기화)
                df_tgt = df_tgt.loc[data_output.index]

                t_name = f"t_{col_name}"
                self.targets.append(t_name)
                data_output[t_name] = df_tgt

                data.drop(columns=[shifted_col], inplace=True)

        # (5) 다음 시점의 `item_id`를 `y_item_id`라는 새로운 타겟 컬럼으로 생성
        data_output[f"y_{self.item_id}"] = data_output.groupby(self.user_id)[
            self.item_id
        ].shift(-1)
        self.targets.append(f"y_{self.item_id}")

        # (6) 최종 컬럼 정리
        columns = (
            [self.user_id, "user_rn", "seq_len", "mask"] + self.features + self.targets
        )
        if self.partition_by is not None:
            columns.append(self.partition_by)

        data_output = (
            data_output[columns]
            .sort_values([self.user_id, "user_rn"])
            .reset_index(drop=True)
        )

        # (7) Null 값이 포함된 행 제거
        # - `shift(-1)` 연산 때문에 마지막 이벤트는 target이 없음(NULL 발생)
        # - 따라서 학습 가능한 행만 추출
        data_output = data_output[data_output.notna().all(axis=1)].reset_index(
            drop=True
        )

        logging.info(
            f"[SequenceGenerator] Skipped {total_skipped} short sequences (< {self.max_seq_len})"
        )

        return data_output


class SequentialDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_sequences: List[str],
        action_sequence: str = None,
        targets: List[str] = None,
        device: str = "cpu",
    ):
        """
        Pandas DataFrame을 torch Dataset으로 변환
        (DataLoader에서 batch 단위 텐서 추출 가능)

        :param df: 시퀀스 데이터셋 (SequenceGenerator 결과)
        :param feature_sequences: 입력 feature로 사용할 컬럼명
        :param action_sequence: 행동 가중치로 사용할 컬럼명
        :param targets: 예측할 target label 컬럼명
        :param device: 텐서를 저장할 장치 (CPU/GPU)
        """

        # {컬럼명: torch.Tensor} 딕셔너리
        self.feature_sequences: Dict[str, torch.Tensor] = {}

        # 마스크 텐서 (패딩 여부: 0=실제값, 1=패딩)
        self.masks: torch.Tensor

        # target label 텐서
        self.targets: Union[Dict[str, torch.Tensor], None] = {}

        # -----------------------------------------------
        # feature 시퀀스 준비
        # -----------------------------------------------
        for feature in feature_sequences:
            if feature == "mask":
                # mask 컬럼: float32 (Transformer attention mask용)
                # shape = (num_samples, max_seq_len)
                x = np.array([np.array(x).astype(np.float32) for x in df[feature]])
                self.masks = torch.from_numpy(x).to(device)
            elif feature == action_sequence:
                continue
            else:
                # 나머지 feature는 정수형 시퀀스 (category_id 등)
                x = np.array([np.array(x).astype(np.int32) for x in df[feature]])
                self.feature_sequences[feature] = torch.from_numpy(x).to(device)

        # -----------------------------------------------
        # 행동 가중치 시퀀스 준비
        # -----------------------------------------------
        if action_sequence is not None:
            x = np.array([np.array(x).astype(np.int32) for x in df[action_sequence]])
            self.action_sequence = torch.from_numpy(x).to(device)
        else:
            self.action_sequence = None

        # -----------------------------------------------
        # target label
        # -----------------------------------------------
        if targets is not None:
            for target in targets:
                # target은 float32로 변환 (CE Loss, BCE Loss 등에서 활용 가능)
                self.targets[target] = torch.from_numpy(
                    np.array([np.array(y).astype(np.float32) for y in df[target]])
                ).to(device)
        else:
            # 추론용 데이터셋에서는 target 없음
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
        개별 샘플 반환 (DataLoader가 batch 단위로 모아줌)
        - feature_sequences: {feature_name: 시퀀스 텐서}
        - mask: 해당 시퀀스의 패딩 마스크
        - targets: target label (있으면 반환, 없으면 None)
        """

        # 입력 feature 시퀀스에서 idx번째 샘플 꺼내기
        feature_sequences = {
            feature_name: sequence[idx]
            for feature_name, sequence in self.feature_sequences.items()
        }

        # target이 있으면 target까지 반환 (학습 및 검증용)
        if self.targets is not None:
            targets = {
                target_name: classes[idx]
                for target_name, classes in self.targets.items()
            }
            # 행동 시퀀스가 있을 때
            if self.action_sequence is not None:
                return (
                    feature_sequences,
                    self.action_sequence[idx],
                    self.masks[idx],
                    targets,
                )
            # 행동 시퀀스가 없을 때
            else:
                return feature_sequences, self.masks[idx], targets

        # target이 없으면 모델 입력 값만 반환 (추론용)
        else:
            if self.action_sequence is not None:
                return feature_sequences, self.action_sequence[idx], self.masks[idx]
            else:
                return feature_sequences, self.masks[idx]


class SequenceVectorDataset(Dataset):
    def __init__(
        self,
        item_id: np.ndarray,
        seq_vector: np.ndarray,
        item_id_to_index: dict,
        device: str,
    ):
        """
        시퀀스 벡터와 매핑된 item index를 반환하는 Dataset
        :param item_id: 원본 item_id 배열
        :param seq_vector: 시퀀스 벡터 배열 (float32)
        :param item_id_to_index: {원본 item_id: index} 매핑 딕셔너리
        :param device: 텐서를 저장할 장치 (CPU/GPU)
        """
        # 원본 item_id를 정수형 index로 변환 (0, 1, 2, ..)
        idx_array = np.array([item_id_to_index[i] for i in item_id], dtype=np.int64)

        # torch Tensor로 변환
        self.item_idx = torch.from_numpy(idx_array).to(device)
        self.seq_vector = torch.from_numpy(seq_vector).to(device)

    def __len__(self):
        """
        전체 데이터 개수 반환
        """
        return len(self.item_idx)

    def __getitem__(self, idx):
        """
        인덱스 idx에 해당하는 샘플 반환
        - (target item index, sequence vector)
        """
        return self.item_idx[idx], self.seq_vector[idx]
