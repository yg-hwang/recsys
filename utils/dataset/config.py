from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DatasetPath:
    """
    데이터셋 저장 경로를 관리하는 Config 클래스
    - dataset_name(예: "fashion")별로 하위 폴더를 자동 생성
    - item & user metadata, user logs, text & image vectors 등 데이터셋 구성 요소별 경로를 프로퍼티로 제공
    """

    # 기본 데이터셋 루트 경로 (기본값: `/tmp/recsys/dataset/`)
    base_dir: Path = field(
        default_factory=lambda: Path("/tmp/recsys/dataset/").resolve()
    )

    # 현재 데이터셋 이름 (예: "fashion")
    dataset_name: str = "fashion"

    def __post_init__(self):
        """
        클래스 초기화 시점에 dataset_name 하위 경로를 생성
        - `/tmp/recsys/dataset/fashion/` (없으면 디렉토리를 새로 생성)
        """
        self._base_path: Path = (self.base_dir / self.dataset_name).resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------
    # 경로 접근자 (property)
    # -----------------------------------------------
    @property
    def base_path(self) -> Path:
        """
        데이터셋 루트 경로 (`/tmp/recsys/dataset/fashion`)
        """
        return self._base_path

    @property
    def item_metadata_path(self) -> Path:
        """
        아이템 메타데이터 저장 경로 (parquet)
        """
        return self.base_path / "item_metadata.parquet"

    @property
    def user_metadata_path(self) -> Path:
        """
        유저 메타데이터 저장 경로 (parquet)
        """
        return self.base_path / "user_metadata.parquet"

    @property
    def user_logs_path(self) -> Path:
        """
        유저-아이템 상호작용(클릭 로그 등) 저장 경로 (디렉토리)
        """
        return self.base_path / "user_logs"

    @property
    def image_vectors_path(self) -> Path:
        """
        아이템 이미지 벡터 저장 경로 (parquet)
        """
        return self.base_path / "image_vectors.parquet"

    @property
    def text_vectors_path(self) -> Path:
        """
        아이템 텍스트 벡터 저장 경로 (parquet)
        """
        return self.base_path / "text_vectors.parquet"

    # -----------------------------------------------
    # 유틸 함수
    # -----------------------------------------------
    def get_path(self, file_name: str) -> Path:
        """
        데이터셋 폴더 내 임의 파일의 경로를 반환
        """
        return self.base_path.joinpath(file_name)

    def create_path(self, file_name: str) -> Path:
        """
        주어진 파일명을 포함한 경로를 생성하고 반환
        - 중간 디렉토리가 없으면 자동 생성
        - 실제 저장 시 안전하게 경로 확보 가능
        """
        path = self.get_path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        return path
