# -----------------------------------------------
# 환경 독립형 sys.path & 한글 폰트 설정
# - Colab, 로컬, 서버 어디서든 동일하게 동작
# -----------------------------------------------
import sys
from pathlib import Path
from matplotlib import font_manager, rcParams


# -----------------------------------------------
# 🔹 1. setup_env가 import될 때 자동으로 recsys 경로 추가
# -----------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    # 선택적으로 로그 출력 가능 (수강생 혼동 방지 위해 False로 유지)
    # print(f"[setup_env] Auto-added to sys.path: {PROJECT_ROOT}")


def setup_path(target_root: str = "recsys", verbose: bool = False) -> Path:
    """
    프로젝트 루트(recsys) 경로를 sys.path에 자동 추가

    :param target_root: 프로젝트 루트 폴더 이름 (기본값: "recsys")
    :param verbose: 경로 추가 로그를 출력할지 여부
    :return: Path 객체 형태의 루트 경로
    """
    root_dir = Path.cwd()

    # 상위 폴더 탐색하면서 recsys 폴더를 찾음
    while root_dir.name != target_root and root_dir.parent != root_dir:
        root_dir = root_dir.parent

    # 이미 sys.path에 추가되어 있지 않으면 append
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
        if verbose:
            print(f"[setup_env] Added to sys.path: {root_dir}")

    return root_dir


def setup_font(
    root_dir: Path, font_path: str = "assets/NanumGothic-Bold.ttf", size: int = 12
) -> None:
    """
    Matplotlib 시각화를 위한 한글 폰트 설정

    :param root_dir: 프로젝트 루트 경로 (`setup_path()` 반환값)
    :param font_path: 폰트 경로
    :param size: 기본 폰트 크기
    """
    font_path = root_dir.joinpath(font_path)

    if not font_path.exists():
        print(f"[ERROR] Font file not found: {font_path}")
        return

    font_name = font_manager.FontProperties(fname=font_path).get_name()

    # matplotlib 전역 폰트 설정
    rcParams["font.family"] = font_name
    rcParams["font.size"] = size
    rcParams["axes.unicode_minus"] = False  # 음수 기호 깨짐 방지

    print(f"[INFO] Font applied: {font_name}")


def setup_env() -> Path:
    """
    전체 환경 설정 (sys.path + 폰트 설정)
    """
    root_dir = setup_path()
    setup_font(root_dir)

    return root_dir
