import json
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Union, List

from .transformer import SimpleTransformer
from .regressor import MultiOutputRegressor


class Model:
    def __init__(
        self,
        seq_model_dir: Union[str, Path],
        reg_model_dir: Union[str, Path],
        padding_value: int = 0,
    ):
        """
        :param seq_model_dir: 트랜스포머 기반 시퀀스 학습 모델
        :param reg_model_dir: 시퀀스 to 상품 벡터 Projection 학습 모델
        :param padding_value: 시퀀스 패딩 값
        """

        self.seq_model_dir = Path(seq_model_dir).resolve()
        self.reg_model_dir = Path(reg_model_dir).resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------------------------
        # Model: Transformer
        # -----------------------------------------------
        # 학습 시 저장해둔 Transformer 설정(config) 로드
        f = open(self.seq_model_dir.joinpath("checkpoint/model_config.json"))
        self.seq_model_config = json.load(f)

        # Transformer 모델 생성 및 가중치 로드
        self.seq_model = SimpleTransformer(**self.seq_model_config)
        self.seq_model.load_state_dict(
            torch.load(
                f=self.seq_model_dir.joinpath("checkpoint/model.pt"),
                map_location=torch.device(self.device),
                weights_only=True,
            )
        )
        self.seq_model = self.seq_model.to(self.device)

        # 패딩 토큰 값 ('-1'을 인코딩 했기 때문에 보통 0임)
        self.padding_value = padding_value

        # feature별 LabelEncoder 로드
        self.encoder = {
            feature: joblib.load(
                self.seq_model_dir.joinpath(f"label_encoders/{feature}.joblib")
            )
            for feature in self.seq_model_config["feature_dims"].keys()
        }

        # -----------------------------------------------
        # Model: Regressor
        # -----------------------------------------------
        # projection 모델 설정(config) 로드
        f = open(self.reg_model_dir.joinpath("checkpoint/model_config.json"))
        self.reg_model_config = json.load(f)

        # projection 모델 생성 및 가중치 로드
        self.reg_model = MultiOutputRegressor(**self.reg_model_config)
        self.reg_model.load_state_dict(
            torch.load(
                f=self.reg_model_dir.joinpath("checkpoint/model.pt"),
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
            "inputs": {
                "brand_name":      ["Myntra", "FILA", "Quiksilver", "Proline"],
                "gender":          ["Men", "Men", "Men", "Men"],
                "age_group":       ["Adults-Men", "Adults-Men", "Adults-Men", "Adults-Men"],
                "base_color":      ["Red", "Navy Blue", "Black", "Red"],
                "season":          ["Summer", "Summer", "Summer", "Summer"],
                "year":            ["2012", "2012", "2012", "2012"],
                "usage":           ["Casual", "Casual", "Casual", "Casual"],
                "master_category": ["Apparel", "Apparel", "Apparel", "Apparel"],
                "sub_category":    ["Topwear", "Topwear", "Topwear", "Topwear"],
                "article_type":    ["Tshirts", "Tshirts", "Tshirts", "Tshirts"],
                "fit":             ["Regular Fit", "Regular Fit", "Regular Fit", "Regular Fit"],
                "occasion":        ["<UNK>", "Casual", "Casual", "Casual"]
            }
        }

        :param body: request body
        :return: body
        """

        inputs = body.get("inputs", {})

        feature_sequences = {}
        for i, (key, values) in enumerate(inputs.items()):
            # -----------------------------------------------
            # feature 정수 인코딩
            # -----------------------------------------------
            seq = []
            for v in values:
                try:
                    # 학습 시 사용된 LabelEncoder로 인코딩
                    v_encoded = self.encoder[key].transform([v]).item()
                except Exception as e:
                    # 학습 시 등장하지 않은 값(unseen)은 "<UNK>"로 대체
                    print(f"`{e} ({key})")
                    v_encoded = self.encoder[key].transform(["<UNK>"]).item()
                seq.append(v_encoded)

            # -----------------------------------------------
            # padding, truncation 적용
            # -----------------------------------------------
            # 부족하면 padding
            if len(seq) < self.seq_model.seq_len:
                seq.extend([self.padding_value] * (self.seq_model.seq_len - len(seq)))
            # 초과하면 truncating
            else:
                seq = seq[: self.seq_model.seq_len]

            # torch Tensor로 변환 (shape: [1, seq_len])
            feature_sequences[key] = (
                torch.from_numpy(np.array(seq, dtype=np.int32))
                .reshape(1, self.seq_model.seq_len)
                .to(self.device)
            )

        input_seq_len = max([len(v) for k, v in inputs.items()])
        masks = [0] * input_seq_len
        # -----------------------------------------------
        # mask도 동일하게 padding, truncation 적용
        # -----------------------------------------------
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
            "user_id": 123,
            "outputs": {...생략...},
            "item_vector": [0.032706137746572495, ..., -0.017184698954224586],
            "seq_vector": [-0.0020149946212768555, ..., 0.08416099101305008],
        }

        :param input_data: request body
        :return: results
        """
        results = list()

        for d in input_data:
            # -----------------------------------------------
            # 모델 입력 형태로 전처리
            # -----------------------------------------------
            data = self.preprocess(body=d)

            # -----------------------------------------------
            # Transformer 기반 예측
            # -----------------------------------------------
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
                logits_flat = y_pred.reshape(-1, dim)

                # Softmax 확률 계산 (seq_len * batch_size, n_classes)
                probs_flat = torch.softmax(logits_flat, dim=-1)

                # argmax로 예측 label 인덱스 추출 (batch_size * seq_len)
                y_pred_ids = logits_flat.argmax(dim=-1)

                # 예측 label의 확률 값 추출
                y_pred_probs = probs_flat[torch.arange(len(y_pred_ids)), y_pred_ids]

                # numpy 변환
                y_pred_ids = y_pred_ids.detach().cpu().numpy()
                y_pred_probs = y_pred_probs.detach().cpu().numpy()

                # label class 복원
                y_pred_labels = self.encoder[target].inverse_transform(y_pred_ids)

                # label class별 확률 평균
                label_probs: Dict[str, List[float]] = {}
                for label, prob in zip(y_pred_labels, y_pred_probs):
                    if label not in label_probs:
                        label_probs[label] = []
                    label_probs[label].append(prob)

                # 평균값으로 단순화
                # - 시퀀스 전체에서 어떤 속성이 얼마나 강하게 예측되는지를 feature별로 요약한 것
                # - 현재 유저가 관심 가질 가능성이 높은 상품 속성을 pre-filter로 사용 가능
                outputs[target] = {
                    str(label): float(np.mean(probs))
                    for label, probs in label_probs.items()
                }
                # ---------- 예시 ---------- #
                # {
                #     "age_group": {
                #         "Adults-Men": 0.9999998211860657,
                #         "Adults-Women": 0.9999995827674866,
                #     },
                #     "article_type": {"Earrings": 0.9999967813491821, "Tshirts": 0.9999957084655762},
                #     "base_color": {
                #         "Green": 0.9999945163726807,
                #         "Navy Blue": 1.0,
                #         "Olive": 0.9999948740005493,
                #         "Red": 0.9999958276748657,
                #         "White": 1.0,
                #     },
                #     "brand_name": {
                #         "ADIDAS": 1.0,
                #         "Adrika": 0.9999954700469971,
                #         "Classic Polo": 0.9999713897705078,
                #         "Lee": 0.9999972581863403,
                #         "Royal Diadem": 0.9999864101409912,
                #     },
                #     "fit": {"<UNK>": 0.9999969601631165, "Regular Fit": 0.9999995231628418},
                #     "gender": {"Men": 0.9999998211860657, "Women": 0.9999999403953552},
                #     "master_category": {
                #         "Accessories": 0.9999992251396179,
                #         "Apparel": 0.9999993443489075,
                #     },

            # -----------------------------------------------
            # 시퀀스 벡터 L2 정규화
            # -----------------------------------------------
            seq_vector = seq_vector.squeeze(0).detach().cpu().numpy()
            seq_vector = seq_vector / np.linalg.norm(seq_vector)

            # -----------------------------------------------
            # projection 모델로 item vector 예측
            # -----------------------------------------------
            with torch.no_grad():
                self.reg_model.eval()
                item_vector = (
                    self.reg_model(
                        torch.from_numpy(seq_vector.astype(np.float32))
                        .to(self.device)
                        .unsqueeze(0)
                    )
                    .squeeze()
                    .detach()
                    .numpy()
                )
                item_vector = item_vector / np.linalg.norm(item_vector)

            # -----------------------------------------------
            # 최종 결과 저장
            # -----------------------------------------------
            results.append(
                {
                    "user_id": data["user_id"],
                    "outputs": outputs,  # feature별 예측 값
                    "seq_vector": seq_vector.tolist(),  # Transformer 시퀀스 벡터
                    "item_vector": item_vector.tolist(),  # Projection 아이템 벡터
                }
            )

        return results
