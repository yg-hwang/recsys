from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DatasetPath:
    """
    데이터셋 저장 경로 클래스
    - `dataset_name`(예: "fashion")별로 하위 폴더를 자동 생성
    - item & user metadata, user logs, text & image vectors 등 데이터셋 구성 요소별 경로를 프로퍼티로 제공
    """

    # 기본 데이터셋 루트 경로 (기본값: `/tmp/recsys/dataset/`)
    root_dir: Path = field(
        default_factory=lambda: Path("/tmp/recsys/dataset/").resolve()
    )

    # 현재 데이터셋 이름 (예: "fashion")
    dataset_name: str = "fashion"

    def __post_init__(self):
        """
        클래스 초기화 시점에 `dataset_name` 하위 경로를 생성
        - 예: `/tmp/recsys/dataset/fashion/` (없으면 디렉토리를 새로 생성)
        """
        self.dataset_path: Path = (self.root_dir.joinpath(self.dataset_name)).resolve()
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------
    # 경로 접근자 (property)
    # -----------------------------------------------
    @property
    def dataset_dir(self) -> Path:
        """
        데이터셋 루트 경로 (`/tmp/recsys/dataset/fashion`)
        """
        return self.dataset_path

    @property
    def item_metadata_path(self) -> Path:
        """
        아이템 메타데이터 저장 경로 (parquet)
        """
        return self.dataset_path.joinpath("item_metadata.parquet")

    @property
    def user_metadata_path(self) -> Path:
        """
        유저 메타데이터 저장 경로 (parquet)
        """
        return self.dataset_path.joinpath("user_metadata.parquet")

    @property
    def user_logs_path(self) -> Path:
        """
        유저-아이템 상호작용(클릭 로그 등) 저장 경로 (디렉토리)
        """
        return self.dataset_path.joinpath("user_logs")

    @property
    def image_vectors_path(self) -> Path:
        """
        아이템 이미지 벡터 저장 경로 (parquet)
        """
        return self.dataset_path.joinpath("image_vectors.parquet")

    @property
    def text_vectors_path(self) -> Path:
        """
        아이템 텍스트 벡터 저장 경로 (parquet)
        """
        return self.dataset_path.joinpath("text_vectors.parquet")
