import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union


def show_items(
    item_ids: List[int],
    df_item: pd.DataFrame,
    image_dir: Union[Path, str],
    attributes: Tuple = (
        "brand_name",
        "master_category",
        "sub_category",
        "article_type",
    ),
    cols: int = 5,
) -> None:
    """
    상품 ID 리스트를 입력받아 이미지 + 상품명 + 속성들을 grid 형태로 시각화

    :param item_ids: 표시할 상품 ID 리스트
    :param df_item: 상품 메타데이터 DataFrame
    :param image_dir: 이미지가 저장된 폴더 경로
    :param attributes: 함께 출력할 속성 컬럼명
    :param cols: Grid 열 (기본 5)
    """
    n_items = len(item_ids)
    rows = math.ceil(n_items / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
    if rows == 1:
        axes = [axes]  # 1행일 때도 반복문 수행
    axes = np.array(axes).reshape(rows, cols)

    for idx, pid in enumerate(item_ids):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        # 해당 상품 정보 가져오기
        item = df_item[df_item["item_id"] == pid].iloc[0]
        item_name = item["name"]

        # 속성 문자열 만들기
        attr_text = "\n".join(
            [f"{attr}: {item[attr]}" for attr in attributes if attr in item]
        )

        # 이미지 경로
        img_path = os.path.join(image_dir, f"{pid}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
        else:
            ax.imshow([[0, 0, 0]])  # 없는 경우 placeholder

        ax.axis("off")
        ax.set_title(f"{item_name}\n{attr_text}", fontsize=10)

    # 남는 빈 칸 처리
    for j in range(n_items, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()
