import json
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Union, List

from .transformer_v2 import MultiTaskMoESequenceTransformer
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
        self.seq_model = MultiTaskMoESequenceTransformer(**self.seq_model_config)
        self.seq_model.load_state_dict(
            torch.load(
                f=self.seq_model_dir.joinpath("checkpoint/model.pt"),
                map_location=torch.device(self.device),
                weights_only=True,
            )
        )
        self.seq_model = self.seq_model.to(self.device)

        # 패딩 토큰 값 ('-999'을 인코딩 했기 때문에 보통 0임)
        self.padding_value = padding_value

        # feature별 LabelEncoder 로드
        self.encoder = {
            feature_name: joblib.load(
                self.seq_model_dir.joinpath(f"label_encoders/{feature_name}.joblib")
            )
            for feature_name in self.seq_model_config["feature_sequence_dims"].keys()
        }
        if "feature_sparse_dims" in self.seq_model_config:
            for feature_name in self.seq_model_config["feature_sparse_dims"].keys():
                self.encoder[feature_name] = joblib.load(
                    self.seq_model_dir.joinpath(f"label_encoders/{feature_name}.joblib")
                )

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
                "feature_sparse": {
                    "user_age": 30,
                    "user_gender": "Men",
                },
                "feature_sequence": {
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
        }

        :param body: request body
        :return: body
        """

        inputs = body.get("inputs", {})

        print(inputs)

        feature_sequence = {}
        for i, (key, values) in enumerate(inputs["feature_sequence"].items()):
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
            feature_sequence[key] = (
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

        feature_sparse = {}
        for k, v in inputs["feature_sparse"].items():
            try:
                # 학습 시 사용된 LabelEncoder로 인코딩
                v_encoded = self.encoder[k].transform([v]).item()
            except Exception as e:
                # 학습 시 등장하지 않은 값(unseen)은 "<UNK>"로 대체
                print(f"`{e} ({k})")
                v_encoded = self.encoder[k].transform(["<UNK>"]).item()
            feature_sparse[k] = torch.tensor(v_encoded).to(self.device).unsqueeze(0)

        # 최종 입력 포맷 구성
        body["inputs"] = {
            "feature_sequence": feature_sequence,
            "masks": masks,
            "feature_sparse": feature_sparse,
        }

        return body

    def predict(self, input_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        예측 수행
        1) `preprocess()`를 통해 모델 입력 형태로 변환
        2) 시퀀스 벡터 + 예측값 출력
        3) 예측값은 LabelEncoder.inverse_transform으로 복원
        4) 시퀀스 벡터를 projection 모델에 입력해 상품 벡터 예측
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
            # 모델 예측
            # -----------------------------------------------
            with torch.no_grad():
                self.seq_model.eval()
                response = self.seq_model(**data["inputs"])
                seq_vector = response["sequence_vector"]
                y_preds = response["y_outputs"]

            # feature별 예측값 처리
            outputs = {
                target: {} for target in self.seq_model_config["output_dims"].keys()
            }
            for target, dim in self.seq_model_config["output_dims"].items():
                # y_pred: (seq_len, batch_size, n_classes)
                y_pred = y_preds[target]

                # logits_flat: (seq_len * batch_size, n_classes)
                logits_flat = y_pred.reshape(-1, dim)

                # Softmax 확률 계산 (seq_len * batch_size, n_classes)
                probs_flat = torch.softmax(logits_flat, dim=-1)

                # Mask 정보 가져오기 (batch_size, seq_len) -> flatten (mask=0: 실제 값, mask=1: 패딩)
                masks_flat = data["inputs"]["masks"].reshape(-1)

                # 유효한 timestep만 선택 (mask == 0)
                # 패딩 위치(mask=1)는 제외하기 위한 boolean mask 생성
                valid_mask = masks_flat == 0

                # 유효한 timestep들의 확률만 추출
                y_probs_valid = probs_flat[valid_mask]

                # 클래스별 평균 확률 (n_classes,)
                # 시퀀스 전체에서 각 클래스가 예측된 평균 확률
                # - 어떤 속성이 얼마나 강하게 예측되는지를 feature별로 요약한 것
                y_probs_mean = y_probs_valid.mean(dim=0).cpu().numpy()

                # Label class 복원
                # 클래스 인덱스를 원래 label 이름으로 변환
                class_indices = np.arange(dim)
                class_labels = self.encoder[target].inverse_transform(class_indices)

                # -----------------------------------------------
                # 예측 Label class 출력
                # -----------------------------------------------
                # 1) 평균 확률 threshold 이상인 class만 후보로 선택
                # - threshold 이하의 노이즈성 예측값 제외
                # - 현재 유저가 관심 가질 가능성이 높은 상품 속성을 pre-filter로 사용 가능
                prob_threshold = 0.02
                candidates = [
                    (str(label), float(p))
                    for label, p in zip(class_labels, y_probs_mean)
                    if p >= prob_threshold
                ]

                # 2) 확률 내림차순 정렬
                # - 가장 높은 확률을 가진 클래스부터 순서대로 정렬
                candidates.sort(key=lambda x: x[1], reverse=True)

                # 3) k-max로 상위 k개만 선택
                # - 과도한 후보 개수를 제한하여 사용성 개선
                k_max = 10
                if len(candidates) > k_max:
                    candidates = candidates[:k_max]

                # 4) threshold가 너무 높아서 아무 것도 안 남은 경우 처리
                # - 최소한 top-1 (가장 높은 확률의 클래스)은 보장
                if not candidates:
                    top_idx = int(y_probs_mean.argmax())
                    candidates = [
                        (str(class_labels[top_idx]), float(y_probs_mean[top_idx]))
                    ]

                # 최종 outputs
                outputs[target] = {label: prob for label, prob in candidates}
                # ---------- 예시 ---------- #
                # {
                #     "age_group": {
                #         "Adults-Women": 0.5079721212387085,
                #         "Adults-Men": 0.2920635938644409,
                #         "Adults-Unisex": 0.1999642252922058
                #     },
                #     "article_type": {
                #         "Flats": 0.38351359963417053,
                #         "Tops": 0.2000497281551361,
                #         "Shirts": 0.2000129222869873,
                #         "Backpacks": 0.19999966025352478
                #     },
                #     ...
                #     "usage": {
                #         "Casual": 0.5993812680244446,
                #         "Ethnic": 0.4000510573387146
                #     },
                #     "year": {
                #         "2012": 0.6900452375411987,
                #         "2015": 0.20011059939861298,
                #         "2016": 0.10978426784276962
                #     }
                # }

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
