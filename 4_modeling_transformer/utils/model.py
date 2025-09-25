import json
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Union, List

from .transformer import SimpleTransformerRec


class Model:
    def __init__(self, model_dir: Union[str, Path], padding_value: int = 0):

        self.model_dir = Path(model_dir).resolve()
        f = open(model_dir.joinpath("checkpoint/model_config.json"))
        self.model_config = json.load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SimpleTransformerRec(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                f=model_dir.joinpath("checkpoint/model.pt"),
                map_location=torch.device(self.device),
                weights_only=True,
            )
        )
        self.model = self.model.to(self.device)
        self.padding_value = padding_value

        self.encoder = {
            feature: joblib.load(model_dir.joinpath(f"label_encoders/{feature}.joblib"))
            for feature in self.model_config["feature_dims"].keys()
        }

    def preprocess(self, body: Dict[str, any]):
        """
        모델 입력값 전처리

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

        for feature in inputs:
            for key, value in feature.items():
                if key not in feature_sequences:
                    feature_sequences[key] = []
                try:
                    value = self.encoder[key].transform([value]).item()
                except Exception as e:
                    print(f"`{e} ({key})")
                    value = self.encoder[key].transform(["NONE"]).item()
                feature_sequences[key].append(value)
            masks.append(0)

        for key in feature_sequences.keys():
            seq = feature_sequences[key]
            if len(seq) < self.model.seq_len:
                seq.extend([self.padding_value] * (self.model.seq_len - len(seq)))
            else:
                seq = seq[: self.model.seq_len]
            feature_sequences[key] = (
                torch.from_numpy(np.array(seq, dtype=np.int32))
                .reshape(1, self.model.seq_len)
                .to(self.device)
            )

        if len(masks) < self.model.seq_len:
            masks.extend([1] * (self.model.seq_len - len(masks)))
        else:
            masks = masks[: self.model.seq_len]
        masks = (
            torch.from_numpy(np.array(masks, dtype=np.float32))
            .reshape(1, self.model.seq_len)
            .to(self.device)
        )

        body["inputs"] = {"feature_sequences": feature_sequences, "masks": masks}

        return body

    def predict(self, input_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        예측 값 및 벡터 반환

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
            data = self.preprocess(body=d)
            outputs = {target: [] for target in self.model_config["output_dims"].keys()}
            with torch.no_grad():
                self.model.eval()
                user_vector, y_preds = self.model(**data["inputs"])

            for target, dim in self.model_config["output_dims"].items():
                y_pred = y_preds[target]
                y_pred = y_pred.permute(1, 0, 2).reshape(-1, dim)
                y_pred = y_pred.argmax(dim=-1)
                y_pred = y_pred.detach().cpu().numpy().tolist()
                y_pred = self.encoder[target].inverse_transform(y_pred)

                outputs[target] = sorted(list(set(y_pred)))

            user_vector = user_vector.squeeze(0).detach().cpu().numpy()
            user_vector = user_vector / np.linalg.norm(user_vector)

            results.append(
                {
                    "user_id": data["user_id"],
                    "outputs": outputs,
                    "user_vector": user_vector.tolist(),
                }
            )

        return results
