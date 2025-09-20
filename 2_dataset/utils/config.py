from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DatasetPath:
    base_dir: Path = field(
        default_factory=lambda: Path("/tmp/recsys/dataset/").resolve()
    )
    dataset_name: str = "fashion"

    def __post_init__(self):
        self._base_path: Path = (self.base_dir / self.dataset_name).resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def item_metadata_path(self) -> Path:
        return self.base_path / "item_metadata.parquet"

    @property
    def user_metadata_path(self) -> Path:
        return self.base_path / "user_metadata.parquet"

    @property
    def interactions_path(self) -> Path:
        return self.base_path / "interactions"

    @property
    def image_vectors_path(self) -> Path:
        return self.base_path / "image_vectors.parquet"

    @property
    def text_vectors_path(self) -> Path:
        return self.base_path / "text_vectors.parquet"

    def get_path(self, file_name: str) -> Path:
        return self.base_path.joinpath(file_name)

    def create_path(self, file_name: str) -> Path:
        path = self.get_path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        return path
