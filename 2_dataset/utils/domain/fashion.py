from typing import List
from dataclasses import dataclass, field


@dataclass
class Fashion:
    color: List[str] = field(
        default_factory=lambda: [
            "화이트",
            "블랙",
            "베이지",
            "네이비",
            "그레이",
            "카키",
            "브라운",
            "아이보리",
            "핑크",
            "라벤더",
            "스카이블루",
            "와인",
            "올리브",
            "민트",
            "오렌지",
            "레드",
        ]
    )
    style: List[str] = field(
        default_factory=lambda: [
            "캐주얼",
            "포멀",
            "스트리트",
            "모던",
            "빈티지",
            "러블리",
            "댄디",
            "스포티",
            "시크",
            "레트로",
            "미니멀",
        ]
    )
    fit: List[str] = field(
        default_factory=lambda: [
            "슬림핏",
            "레귤러핏",
            "오버핏",
            "크롭핏",
            "루즈핏",
            "세미오버핏",
        ]
    )
    material: List[str] = field(
        default_factory=lambda: [
            "코튼",
            "폴리",
            "데님",
            "린넨",
            "울",
            "나일론",
            "스웨이드",
            "플리스",
            "퍼",
            "가죽",
            "메쉬",
            "레이온",
            "모달",
            "아크릴",
        ]
    )
    season: List[str] = field(
        default_factory=lambda: ["봄", "여름", "가을", "겨울", "간절기"]
    )
    sleeve: List[str] = field(
        default_factory=lambda: ["민소매", "숏", "롱", "7부", "숏슬리브", "롱슬리브"]
    )
    category: List[str] = field(
        default_factory=lambda: [
            "셔츠",
            "블라우스",
            "티셔츠",
            "니트",
            "후드티",
            "맨투맨",
            "재킷",
            "점퍼",
            "코트",
            "조끼",
            "베스트",
            "청바지",
            "슬랙스",
            "와이드팬츠",
            "조거팬츠",
            "스커트",
            "원피스",
            "오버롤",
            "트레이닝세트",
            "레깅스",
            "카디건",
        ]
    )
    name_templates: List[str] = field(
        default_factory=lambda: [
            "{season} {fit} {color} {material} {sleeve} {category}",
            "{style} 무드의 {color} {category}",
            "{fit} 실루엣의 {material} {category}",
            "{season}에 어울리는 {color} {category}",
            "{season} 시즌, {style} 무드의 {color} {material} {sleeve} {fit} {category}",
            "{season} 한정 {material} 소재 {color} {category}",
            "트렌디한 {style}룩, {fit} {category} in {season}",
            "{color} 컬러의 {style} 스타일 {material} {category}",
            "{season} 감성 {fit} {material} {category}",
            "필수템! {style} 무드의 {season}용 {category}",
        ]
    )
