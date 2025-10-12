#!/bin/bash
# Milvus Standalone 자동 설치 및 실행 스크립트
# version: 2.4.6

# -----------------------------------------------
# 1. Docker Compose 파일 다운로드
# -----------------------------------------------
echo ">>> Downloading Milvus Standalone docker-compose.yml ..."
wget -q https://github.com/milvus-io/milvus/releases/download/v2.4.6/milvus-standalone-docker-compose.yml -O docker-compose.yml

# -----------------------------------------------
# 2. 볼륨 디렉토리 생성
# -----------------------------------------------
echo ">>> Creating volume directories (./volumes/*)"
mkdir -p volumes/milvus volumes/minio volumes/etcd

# -----------------------------------------------
# 3. 컨테이너 실행
# -----------------------------------------------
echo ">>> Starting Milvus containers..."
docker compose up -d

# -----------------------------------------------
# 4. 상태 확인
# -----------------------------------------------
echo ">>> Checking running containers..."
docker ps --filter "name=milvus"

# -----------------------------------------------
# 5. 완료 메시지
# -----------------------------------------------
echo "Milvus Standalone is running!"
echo "   - Host: localhost"
echo "   - Port: 19530 (for PyMilvus)"
echo "   - Dashboard: http://localhost:9091"
echo "   - Data path: ./volumes/milvus/"