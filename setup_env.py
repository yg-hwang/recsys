# -----------------------------------------------
# 환경 독립형 sys.path & 한글 폰트 설정
# - Colab, 로컬, 서버 어디서든 동일하게 동작
# -----------------------------------------------
import sys
from pathlib import Path
from matplotlib import font_manager, rcParams


# -----------------------------------------------
# import할 때 자동으로 `recsys` 경로 추가
# -----------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


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
    root_dir: Path,
    font_path: str = "assets/NanumGothic-Bold.ttf",
    size: int = 12,
    verbose: bool = True,
) -> None:
    """
    Matplotlib 시각화를 위한 한글 폰트 설정

    :param root_dir: 프로젝트 루트 경로 (`setup_path()` 반환값)
    :param font_path: 폰트 경로
    :param size: 기본 폰트 크기
    :param verbose: 폰트 적용 경로 출력 (선택)
    """
    font_file = root_dir.joinpath(font_path)
    if not font_file.exists():
        print(f"[ERROR] Font not found: {font_file}")
        return

    # 1) 명시적으로 matplotlib에 폰트 파일 등록
    font_manager.fontManager.addfont(str(font_file))

    # 2) 내부 이름 읽기
    prop = font_manager.FontProperties(fname=str(font_file))
    font_name = prop.get_name()

    # 3) 폰트 목록에 반영되었는지 확인, 필요시 재빌드
    if font_name not in {f.name for f in font_manager.fontManager.ttflist}:
        try:
            font_manager._rebuild()
        except Exception as e:
            if verbose:
                print("[INFO] rebuild warning:", e)

    # 4) rcParams에 안전하게 적용
    rcParams["font.family"] = "sans-serif"
    current_sans = list(rcParams.get("font.sans-serif", []))
    rcParams["font.sans-serif"] = [font_name] + [
        s for s in current_sans if s != font_name
    ]
    rcParams["font.size"] = size
    rcParams["axes.unicode_minus"] = False


def setup_env() -> Path:
    """
    전체 환경 설정 (sys.path + 폰트 설정)
    """
    root_dir = setup_path()
    setup_font(root_dir)

    return root_dir
