import os
import bentoml
from bentoml.io import JSON
from pathlib import Path
from typing import List, Dict

from utils.model import Model

DOMAIN = "fashion"
current_dir = os.path.abspath(os.curdir)
base_dir = "/".join(current_dir.split("/")[:-1])
model_dir = Path(base_dir).joinpath(f"data/model/{DOMAIN}")
model_dir.mkdir(parents=True, exist_ok=True)
model = Model(model_dir=model_dir)


svc = bentoml.Service("rec_service")


@svc.api(input=JSON(), output=JSON())
def predict(input_data: List[Dict[str, any]]) -> dict:
    results = model.predict(input_data)
    return {"predictions": results}
