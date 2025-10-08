# 추천 시스템 데모 만들기  
> LightGCN + TransformerEncoder + ANN 기반 추천 시스템 구축 & BentoML 모델 서빙

---

## 개요

이 프로젝트는 **추천 시스템 강의 실습용 데모 시스템**으로,  
사용자 행동 로그를 기반으로 임베딩을 학습하고  
LightGCN, TransformerEncoder, ANN을 활용하여 **실시간 추천 파이프라인**을 구현합니다.

---

## 핵심 구조

| 단계 | 내용 |
|------|------|
| **1️⃣ LightGCN 기반 User·Item 임베딩** | 사용자–상품 상호작용 그래프에서 임베딩 학습 |
| **2️⃣ TransformerEncoder 기반 seq-to-seq 모델링** | 사용자 시퀀스를 입력받아 Multi-task 방식으로 시퀀스 벡터 생성 |
| **3️⃣ ANN 기반 후보군 검색 (Retrieval)** | LightGCN 기반 ANN, Transformer 기반 ANN 각각으로 추천 후보군 생성 |
| **4️⃣ 모델 서빙 (BentoML)** | 학습된 모델을 BentoML로 패키징 후 REST API 형태로 배포 |

---

## 개발 환경

| 항목 | 권장 버전 |
|------|-----------|
| Python | 3.10 ~ 3.12 |
| PyTorch | >= 2.2 |
| CUDA Toolkit | >= 11.8 |
| OS | macOS / Windows / Linux / Google Colab |
| RAM | 8GB 이상 (GPU VRAM 4GB 이상 권장) |