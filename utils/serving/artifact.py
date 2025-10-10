import shutil
import bentoml
import tempfile
from pathlib import Path
from typing import Union, Optional


def register_artifact_model(
    model_dir: Union[str, Path],
    name: str,
    api_version: str = "v1",
    module_name: str = "custom.artifact_bundle",
    metadata: Optional[dict] = None,
    verbose: bool = False,
) -> bentoml.Model:
    """
    폴더 단위 BentoML 모델 등록 함수
    - 학습 완료된 모델 산출물 폴더 전체를 BentoML 저장소에 버전 단위로 등록
    - model.pt, config.json, label_encoders/*.joblib 등 여러 파일을 통째로 관리
    - BentoML 모델 버전 관리(tag) 및 metadata 기록 지원

    Example:
    register_artifact_model(
        model_dir="data/outputs/fashion/transformer",
        name="transformer",
        metadata={"domain": "fashion", "description": "Transformer artifact folder"}
    )

    :param model_dir: 학습 완료된 모델 아티팩트 폴더 경로
    :param name: BentoML 저장소에 등록할 모델 이름
    :param api_version: API 버전 태그 (기본값: "v1")
    :param module_name: 내부 module 이름 (표시용)
    :param metadata: 모델에 부가 설명 정보 (dict)
    :param verbose: 저장 경로 출력
    :return: 저장된 bentoml.Model 객체
    """

    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: `{model_dir}`")

    # 1) 임시 복사본 생성
    temp_dir = Path(tempfile.mkdtemp())
    artifact_dir = temp_dir.joinpath("model_artifact")
    shutil.copytree(model_dir, artifact_dir, dirs_exist_ok=True)

    # 2) BentoML 모델 생성 컨텍스트
    with bentoml.models.create(
        name=name,
        module=module_name,
        api_version=api_version,
        metadata=metadata or {},
    ) as bento_model:
        # 3) 모델 저장소 내부에 아티팩트 전체 복사
        target_dir = Path(bento_model.path_of("artifact"))
        shutil.copytree(artifact_dir, target_dir, dirs_exist_ok=True)

    if verbose:
        print(f"Completed to registry BentoML: `{bento_model.tag}`")
        print(f"Artifact path: `{bento_model.path_of('artifact')}`")

    return bento_model
