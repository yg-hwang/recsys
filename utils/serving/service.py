import os
import sys
from pathlib import Path
from typing import List, Dict

import bentoml
from bentoml.io import JSON

# project_root 기준 경로 자동 계산
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.model_transformer.model import Model as Transformer
from utils.model_lightgcn.model import Model as LightGCN


DOMAIN = "fashion"
current_dir = os.path.abspath(os.curdir)
base_dir = "/".join(current_dir.split("/")[:-2])
model_dir = Path(base_dir).joinpath(f"data/model/{DOMAIN}")

model_transformer = Transformer(model_dir=model_dir)
model_lightgcn = LightGCN(model_dir=model_dir)

svc = bentoml.Service("rec_service")


# ---------- API ---------- #
@svc.api(input=JSON(), output=JSON())
def predict_lightgcn(input_data: List[Dict[str, any]]) -> dict:
    return {"predictions": model_lightgcn.predict(input_data)}


@svc.api(input=JSON(), output=JSON())
def predict_transformer(input_data: List[Dict[str, any]]) -> dict:
    return {"predictions": model_transformer.predict(input_data)}


# ---------- Reload ---------- #
# 새 checkpoint 파일이 덮어씌워졌을 때 호출하면 메모리에 올린 모델을 다시 불러온다.#
@svc.api(input=JSON(), output=JSON())
def reload_model_lightgcn(_: dict) -> dict:

    global model_lightgcn
    model_lightgcn = LightGCN(model_dir=model_dir)
    return {"status": "ok", "message": "Model reloaded successfully."}


@svc.api(input=JSON(), output=JSON())
def reload_model_transformer(_: dict) -> dict:
    global model_transformer
    model_transformer = Transformer(model_dir=model_dir)
    return {"status": "ok", "message": "Model reloaded successfully."}
