import json
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Union, List

from .transformer import SimpleTransformer
from .regressor import MultiOutputRegressor


class Model:
    def __init__(self, model_dir: Union[str, Path], padding_value: int = 0):

        self.model_dir = Path(model_dir).resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------------------------
        # Model: Transformer
        # -----------------------------------------------
        model = "transformer"

        # 학습 시 저장해둔 Transformer 설정(config) 로드
        f = open(self.model_dir.joinpath(f"{model}/checkpoint/model_config.json"))
        self.seq_model_config = json.load(f)

        # Transformer 모델 생성 및 가중치 로드
        self.seq_model = SimpleTransformer(**self.seq_model_config)
        self.seq_model.load_state_dict(
            torch.load(
                f=model_dir.joinpath(f"{model}/checkpoint/model.pt"),
                map_location=torch.device(self.device),
                weights_only=True,
            )
        )
        self.seq_model = self.seq_model.to(self.device)

        # 패딩 토큰 값 (보통 0)
        self.padding_value = padding_value

        # feature별 LabelEncoder 로드
        self.encoder = {
            feature: joblib.load(
                model_dir.joinpath(f"{model}/label_encoders/{feature}.joblib")
            )
            for feature in self.seq_model_config["feature_dims"].keys()
        }

        # -----------------------------------------------
        # Model: Regressor
        # -----------------------------------------------
        model = "regressor"

        # projection 모델 설정(config) 로드
        f = open(self.model_dir.joinpath(f"{model}/checkpoint/model_config.json"))
        self.reg_model_config = json.load(f)

        # projection 모델 생성 및 가중치 로드
        self.reg_model = MultiOutputRegressor(**self.reg_model_config)
        self.reg_model.load_state_dict(
            torch.load(
                f=model_dir.joinpath(f"{model}/checkpoint/model.pt"),
                map_location=torch.device(self.device),
                weights_only=True,
            )
        )
        self.reg_model = self.reg_model.to(self.device)

    def preprocess(self, body: Dict[str, any]) -> Dict[str, any]:
        """
        입력 request body를 모델 입력 형식으로 변환
        - feature 값들을 LabelEncoder로 정수 인코딩
        - 시퀀스 길이를 맞추기 위해 padding 수행
        - mask 생성 (0=실제 값, 1=패딩)

        ---------- 예시 입력 ----------
        {
            "user_id": 123,
            "inputs": [
                {
                    "color": "화이트",
                    "style": "캐주얼",
                    "fit": "레귤러핏",
                    "material": "코튼",
                    "season": "가을",
                    "sleeve": "롱",
                    "category": "블라우스",
                },
                {
                    "color": "그레이",
                    "style": "포멀",
                    "fit": "오버핏",
                    "material": "퍼",
                    "season": "가을",
                    "sleeve": "롱",
                    "category": "점퍼",
                },
            ],
        }

        :param body: request body
        :return: body
        """

        inputs = body.get("inputs", [])
        feature_sequences = {}
        masks = []

        # feature 값 -> 정수 인코딩
        for feature in inputs:
            for key, value in feature.items():
                if key not in feature_sequences:
                    feature_sequences[key] = []
                try:
                    # 학습 시 사용된 LabelEncoder로 인코딩
                    value = self.encoder[key].transform([value]).item()
                except Exception as e:
                    # 학습 시 등장하지 않은 값(unseen)은 "NONE"으로 대체
                    print(f"`{e} ({key})")
                    value = self.encoder[key].transform(["NONE"]).item()
                feature_sequences[key].append(value)
            masks.append(0)  # 실제 토큰 값(mask=0)

        # padding, truncation 적용
        for key in feature_sequences.keys():
            seq = feature_sequences[key]
            # 부족하면 padding 추가
            if len(seq) < self.seq_model.seq_len:
                seq.extend([self.padding_value] * (self.seq_model.seq_len - len(seq)))
            # 초과하면 잘라냄
            else:
                seq = seq[: self.seq_model.seq_len]

            # torch Tensor로 변환 (shape: [1, seq_len])
            feature_sequences[key] = (
                torch.from_numpy(np.array(seq, dtype=np.int32))
                .reshape(1, self.seq_model.seq_len)
                .to(self.device)
            )

        # mask도 동일하게 padding, truncation 적용
        if len(masks) < self.seq_model.seq_len:
            masks.extend([1] * (self.seq_model.seq_len - len(masks)))
        else:
            masks = masks[: self.seq_model.seq_len]
        masks = (
            torch.from_numpy(np.array(masks, dtype=np.float32))
            .reshape(1, self.seq_model.seq_len)
            .to(self.device)
        )

        # 최종 입력 포맷 구성
        body["inputs"] = {"feature_sequences": feature_sequences, "masks": masks}

        return body

    def predict(self, input_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        예측 수행
        1) `preprocess()`를 통해 Transformer 입력 형태로 변환
        2) 시퀀스 벡터 + feature별 예측값 출력
        3) feature별 예측값은 LabelEncoder.inverse_transform으로 복원
        4) 시퀀스 벡터를 projection 모델에 입력해 item 벡터 추출
        5) 최종 결과 반환

        ---------- 출력 예시 ----------
        {
            "color": ["그레이", "화이트"],
            "style": ["레트로", "빈티지", "캐주얼", "포멀"],
            "fit": ["레귤러핏", "루즈핏", "오버핏"],
            "material": ["코튼", "퍼"],
            "season": ["가을", "여름"],
            "sleeve": ["롱", "롱슬리브"],
            "category": ["블라우스", "셔츠", "점퍼"],
        }

        :param input_data: request body
        :return: results
        """
        results = list()

        for d in input_data:
            # 모델 입력 형태로 전처리
            data = self.preprocess(body=d)

            # Transformer 예측
            outputs = {
                target: {} for target in self.seq_model_config["output_dims"].keys()
            }
            with torch.no_grad():
                self.seq_model.eval()
                seq_vector, y_preds = self.seq_model(**data["inputs"])

            # feature별 예측값 처리
            for target, dim in self.seq_model_config["output_dims"].items():
                # y_pred: (seq_len, batch_size, n_classes)
                y_pred = y_preds[target]

                # logits_flat: (seq_len * batch_size, n_classes)
                logits_flat = y_pred.permute(1, 0, 2).reshape(-1, dim)

                # Softmax 확률 계산 (seq_len * batch_size, n_classes)
                probs_flat = torch.softmax(logits_flat, dim=-1)

                # argmax로 예측 클래스 인덱스
                y_pred_ids = logits_flat.argmax(dim=-1)

                # 예측 클래스의 확률 값 추출
                y_pred_probs = probs_flat[torch.arange(len(y_pred_ids)), y_pred_ids]

                # numpy 변환
                y_pred_ids = y_pred_ids.detach().cpu().numpy()
                y_pred_probs = y_pred_probs.detach().cpu().numpy()

                # 라벨 복원
                y_pred_labels = self.encoder[target].inverse_transform(y_pred_ids)

                # -----------------------------------------------
                # 클래스별 확률 평균
                # -----------------------------------------------
                label_probs: Dict[str, List[float]] = {}
                for label, prob in zip(y_pred_labels, y_pred_probs):
                    if label not in label_probs:
                        label_probs[label] = []
                    label_probs[label].append(prob)

                # 평균값으로 단순화
                outputs[target] = {
                    label: float(np.mean(probs)) for label, probs in label_probs.items()
                }
                # ---------- 예시 ---------- #
                # {
                #     "category": {
                #         "베스트": 0.9591713547706604,
                #         "블라우스": 0.9999942779541016,
                #         "점퍼": 0.9999899864196777
                #     },
                #     "color": {
                #         "그레이": 0.9999945163726807,
                #         "네이비": 0.9791531562805176,
                #         "화이트": 0.9999948740005493
                #     },
                #     "fit": {
                #         "레귤러핏": 0.9999972581863403,
                #         "루즈핏": 0.7540665864944458,
                #         "오버핏": 0.9999918937683105
                #     },
                #     "material": {
                #         "레이온": 0.9347188472747803,
                #         "코튼": 0.9999949932098389,
                #         "퍼": 0.9999948740005493
                #     },
                #     "season": {"가을": 0.9999992251396179, "간절기": 0.6271464228630066},
                #     "sleeve": {"7부": 0.8122173547744751, "롱": 0.9999983310699463}
                #     "style": {
                #         "빈티지": 0.5657854080200195,
                #         "캐주얼": 0.9999991655349731,
                #         "포멀": 0.9999949932098389
                #     },
                # }

            # 시퀀스 벡터 L2 정규화
            seq_vector = seq_vector.squeeze(0).detach().cpu().numpy()
            seq_vector = seq_vector / np.linalg.norm(seq_vector)

            # projection 모델로 item vector 예측
            with torch.no_grad():
                self.reg_model.eval()
                item_vector = (
                    self.reg_model(
                        torch.from_numpy(seq_vector).to(self.device).unsqueeze(0)
                    )
                    .squeeze()
                    .detach()
                    .numpy()
                )

            # 최종 결과 저장
            results.append(
                {
                    "user_id": data["user_id"],
                    "outputs": outputs,  # feature별 예측 값
                    "seq_vector": seq_vector.tolist(),  # Transformer 시퀀스 벡터
                    "item_vector": item_vector.tolist(),  # Projection 아이템 벡터
                }
            )

        return results
