import sys
from pathlib import Path
from typing import Dict

import bentoml
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException

# -----------------------------------------------
# 프로젝트 경로 설정
# -----------------------------------------------
# setup_env를 import하기 위해 루트 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[2]))
from setup_env import setup_path

# recsys 루트를 sys.path에 추가
root_dir = setup_path()

# -----------------------------------------------
# 모델 불러오기
# -----------------------------------------------
from models.lightgcn.model import Model as LightGCN
from models.transformer.model import Model as Transformer


# -----------------------------------------------
# BentoML 서비스 정의 및 모델 로딩
# -----------------------------------------------
class RecommendationService:
    """
    추천 서비스 클래스
    """

    def __init__(self):
        self.model_transformer: Transformer = None
        self.model_lightgcn: LightGCN = None

    @bentoml.on_startup
    def load_models(self):
        """
        서비스 시작 시 모델을 메모리에 로드
        """

        print("---------- 모델 로딩 시작 ----------")
        try:
            # 1. Transformer 모델 로드
            seq_model_artifact = bentoml.models.get("transformer:latest")
            reg_model_artifact = bentoml.models.get("regressor:latest")
            seq_model_dir = Path(seq_model_artifact.path_of("artifact"))
            reg_model_dir = Path(reg_model_artifact.path_of("artifact"))

            self.model_transformer = Transformer(
                seq_model_dir=seq_model_dir, reg_model_dir=reg_model_dir
            )

            # 2. LightGCN 모델 로드
            lightgcn_model_artifact = bentoml.models.get("lightgcn:latest")
            lightgcn_model_dir = Path(lightgcn_model_artifact.path_of("artifact"))
            self.model_lightgcn = LightGCN(model_dir=lightgcn_model_dir)

            print("---------- 모델 로딩 완료 ----------")

        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
            raise BentoMLException(f"모델 로딩 실패: {e}")

    def predict_lightgcn(self, body: Dict[str, any]) -> dict:
        """
        LightGCN 모델 기반 추천 상품
        - 입력: 유저 ID
        - 출력: 추천 결과

        body 예시:
        {
            "input_data": [{"user_id": 123}, {"user_id": 456}],
            "top_k": 50
        }
        """
        if not self.model_lightgcn:
            # 모델이 로드되지 않았을 경우를 대비한 방어 코드
            raise BentoMLException("LightGCN 모델이 로드되지 않았습니다.")

        input_data = body.get("input_data", [])
        top_k = body.get("top_k", 100)

        return {
            "predictions": self.model_lightgcn.predict(
                input_data=input_data, top_k=top_k
            )
        }

    def predict_transformer(self, body: Dict[str, any]) -> dict:
        """
        Transformer 기반 시퀀스 모델 예측
        - 입력: Feature Sequence
        - 출력: 예측된 label class 후보, seq_vector, item_vector
        """

        if not self.model_transformer:
            raise BentoMLException("Transformer 모델이 로드되지 않았습니다.")

        input_data = body.get("input_data", [])

        return {"predictions": self.model_transformer.predict(input_data=input_data)}


# -----------------------------------------------
# BentoML 서비스 인스턴스 정의 및 API 등록
# -----------------------------------------------
# 서비스 인스턴스 생성
svc = bentoml.Service(
    "rec_service",
    models=[
        bentoml.models.get("transformer:latest"),
        bentoml.models.get("regressor:latest"),
        bentoml.models.get("lightgcn:latest"),
    ],
)

rec_service_instance = RecommendationService()
svc.on_startup(lambda app: rec_service_instance.load_models())


# -----------------------------------------------
# API 엔드포인트 정의
# -----------------------------------------------
@svc.api(input=JSON(), output=JSON())
def predict_lightgcn(body: Dict[str, any]) -> dict:
    """
    LightGCN 모델 기반 추천 API
    """
    return rec_service_instance.predict_lightgcn(body)


@svc.api(input=JSON(), output=JSON())
def predict_transformer(body: Dict[str, any]) -> dict:
    """
    Transformer 기반 시퀀스 추천 API
    """
    return rec_service_instance.predict_transformer(body)
