import sys
from pathlib import Path
from typing import List, Dict

import bentoml
from bentoml.io import JSON

# -----------------------------------------------
# 프로젝트 경로 설정
# -----------------------------------------------
# setup_env를 import하기 위해 루트 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[2]))
from setup_env import setup_path

root_dir = setup_path()  # recsys 루트를 sys.path에 추가

# -----------------------------------------------
# 모델 불러오기
# -----------------------------------------------
# - Transformer: 시퀀스 기반 추천 모델 (Sequential Transformer + Projection)
# - LightGCN: 협업 필터링 기반 추천 모델
from models.transformer.model import Model as Transformer
from models.lightgcn.model import Model as LightGCN

# 모델 저장 디렉토리 경로
model_dir = Path(root_dir).joinpath("data/outputs/fashion")

# 실제 모델 객체 메모리에 로드
model_transformer = Transformer(model_dir=model_dir)
model_lightgcn = LightGCN(model_dir=model_dir)

# -----------------------------------------------
# BentoML 서비스 정의
# -----------------------------------------------
# - 서비스 이름은 "rec_service"
svc = bentoml.Service("rec_service")


# -----------------------------------------------
# API 엔드포인트 정의
# -----------------------------------------------
@svc.api(input=JSON(), output=JSON())
def predict_lightgcn(input_data: List[Dict[str, any]]) -> dict:
    """
    LightGCN 모델 기반 추천 아이템 제공 API
    - 입력: JSON (유저 ID)
    - 출력: JSON (추천 결과)
    """
    return {"predictions": model_lightgcn.predict(input_data)}


@svc.api(input=JSON(), output=JSON())
def predict_transformer(input_data: List[Dict[str, any]]) -> dict:
    """
    Transformer 기반 시퀀스 추천 아이템 제공 API
    - 입력: JSON (Feature Sequence)
    - 출력: JSON (예측된 feature 후보, seq_vector, item_vector)
    """
    return {"predictions": model_transformer.predict(input_data)}


# -----------------------------------------------
# 모델 Reload API
# -----------------------------------------------
# - 새로운 checkpoint가 저장되었을 때 호출하면 메모리에 올려둔 모델을 다시 로드하여 반영
@svc.api(input=JSON(), output=JSON())
def reload_model_lightgcn(_: dict) -> dict:
    """
    LightGCN 모델 리로드
    - 입력: JSON (내용 무관)
    - 출력: 상태 메시지

    ---------- CLI 예시 ----------
    # 서비스가 3000번 포트에서 실행 중일 때
    `curl -X POST http://localhost:3000/reload_model_lightgcn -H "Content-Type: application/json" -d '{}'`
    """
    global model_lightgcn
    model_lightgcn = LightGCN(model_dir=model_dir)
    return {"status": "ok", "message": "Model reloaded successfully."}


@svc.api(input=JSON(), output=JSON())
def reload_model_transformer(_: dict) -> dict:
    """
    Transformer 모델 리로드
    - 입력: 아무 JSON (내용 무관)
    - 출력: 상태 메시지

    ---------- CLI 예시 ----------
    # 서비스가 3000번 포트에서 실행 중일 때
    `curl -X POST http://localhost:3000/reload_model_transformer -H "Content-Type: application/json" -d '{}'`
    """
    global model_transformer
    model_transformer = Transformer(model_dir=model_dir)
    return {"status": "ok", "message": "Model reloaded successfully."}
