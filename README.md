# Recommendation System Demo

## Overview

| 단계                                          | 내용                                                                    |
|---------------------------------------------|-----------------------------------------------------------------------|
| **1) LightGCN 기반 User·Item 임베딩**            | 사용자–상품 상호작용 그래프에서 임베딩 학습                                              |
| **2) TransformerEncoder 기반 seq-to-seq 모델링** | 사용자 시퀀스를 입력받아 Multi-task 방식으로 시퀀스 벡터 생성                               |
| **3) 모델 서빙 (BentoML)**                      | 학습된 모델을 BentoML로 패키징 후 REST API 형태로 배포                                |
| **4) Vector Search (Retrieval)**            | Vector DB를 활용하여 추천 후보군 생성                                             |
| **5) Ranking Scoring (Ranker)**             | CTR 예측 모델 기반의 상품 정렬 |

## Development Environment
- Python (3.12)
- Torch
- HNSWlib
- Milvus
- Docker
- Streamlit
- etc.