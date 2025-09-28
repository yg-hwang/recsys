import random
import pandas as pd
from typing import Dict, Type
from dataclasses import fields


class ProductNameGenerator:
    """
    도메인 클래스(Fashion 등) 기반 아이템 제목 생성기
    - 도메인에 정의된 속성값들을 무작위로 선택
    - `name_templates` 중 하나를 선택해 속성들을 치환하여 title 생성
    """

    def __init__(self, config):
        self.config = config  # Fashion 같은 도메인 클래스 인스턴스

    def generate(self) -> Dict[str, str]:
        """
        하나의 아이템(title + 속성 dict)을 생성
        :return: {"title": 아이템 제목, "color": ..., "style": ..., ...}
        """

        # 1) 아이템 제목 템플릿 중 하나 랜덤 선택
        template = random.choice(self.config.name_templates)

        # 2) 도메인 속성값에서 무작위 선택 -> 템플릿 치환 context 구성
        context = {}
        for field_obj in fields(self.config):  # dataclass의 모든 필드 확인
            field_name = field_obj.name
            if field_name == "name_templates":
                continue  # 템플릿은 제외
            values = getattr(self.config, field_name)
            context[field_name] = random.choice(values)

        # 3) 템플릿에 속성값 대입 -> 최종 아이템 제목 생성
        title = template.format(**context)

        # 4) 결과 반환 (아이템 제목 + 속성 dict)
        return {"title": title, **context}


def generate_items(domain_class: Type, num_items: int = 10000) -> pd.DataFrame:
    """
    도메인 클래스 기반으로 아이템 메타데이터를 생성하고 저장하는 함수

    :param domain_class: 도메인 클래스 (예: Fashion, Food, Book 등)
    :param num_items: 생성할 아이템 수
    :return: 아이템 메타데이터 DataFrame (item_id, title, 속성...)
    """

    # 1) 도메인 클래스 인스턴스 생성 (예: Fashion())
    domain_instance = domain_class()

    # 2) 아이템 제목 생성기 초기화
    generator = ProductNameGenerator(domain_instance)

    # 3) 데이터 생성
    records = []
    seen = set()  # 중복 title 방지용

    for _ in range(num_items):
        for _ in range(10):  # 최대 10회 재시도 (중복 title 회피)
            rec = generator.generate()
            t = rec["title"]
            if t not in seen:
                seen.add(t)
                records.append(rec)
                break
        else:
            # 재시도 끝에도 중복이면 그대로 추가(중복 허용)
            records.append(rec)

    # 4) DataFrame 변환 + `item_id` 추가
    df_result = pd.DataFrame(records)
    df_result.insert(0, "item_id", df_result.index + 1)

    return df_result
