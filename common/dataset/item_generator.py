import random
import pandas as pd
from typing import Dict, Type
from dataclasses import fields


class ProductNameGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self) -> Dict[str, str]:
        template = random.choice(self.config.name_templates)

        context = {}
        for field_obj in fields(self.config):
            field_name = field_obj.name
            if field_name == "name_templates":
                continue
            values = getattr(self.config, field_name)
            context[field_name] = random.choice(values)

        title = template.format(**context)

        return {"title": title, **context}


def generate_items(domain_class: Type, num_items: int = 10000) -> pd.DataFrame:
    """
    도메인 클래스 기반으로 상품 메타데이터를 생성하고 저장하는 함수

    :param domain_class: 도메인 클래스 (예: Fashion, Food, Book 등)
    :param num_items: 생성할 상품 수
    """

    # 도메인 인스턴스 생성 및 생성기 초기화
    domain_instance = domain_class()
    generator = ProductNameGenerator(domain_instance)

    # 데이터 생성
    # records = [generator.generate() for _ in range(num_items)]

    # 데이터 생성 (중복 title 방지 시도, 실패하면 중복 허용)
    records = []
    seen = set()
    for _ in range(num_items):
        for _ in range(10):  # 최대 10회 재시도
            rec = generator.generate()
            t = rec["title"]
            if t not in seen:
                seen.add(t)
                records.append(rec)
                break
        else:
            # 재시도 끝에도 중복이면 그대로 추가(중복 허용)
            records.append(rec)

    df_result = pd.DataFrame(records)
    df_result.insert(0, "item_id", df_result.index + 1)

    return df_result
