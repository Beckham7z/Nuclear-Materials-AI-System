#!/bin/bash

set -e

echo "🧹 清理已存在的容器..."
sudo docker stop mongodb neo4j milvus-standalone milvus-etcd milvus-minio 2>/dev/null || true
sudo docker rm mongodb neo4j milvus-standalone milvus-etcd milvus-minio 2>/dev/null || true

echo "📥 拉取镜像..."
sudo docker pull mongo:latest
sudo docker pull neo4j:latest
sudo docker pull milvusdb/milvus:v2.3.4
sudo docker pull minio/minio:RELEASE.2023-03-20T20-16-18Z
sudo docker pull quay.io/coreos/etcd:v3.5.5

echo "�� 启动 MongoDB..."
sudo docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=root \
  -e MONGO_INITDB_ROOT_PASSWORD=root \
  -v mongodb_data:/data/db \
  mongo:latest

echo "🚀 启动 Neo4j..."
sudo docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/1q2w3e4r \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  -v neo4j_import:/var/lib/neo4j/import \
  neo4j:latest

echo "🚀 启动 Milvus 组件..."
# 启动 etcd
sudo docker run -d \
  --name milvus-etcd \
  -p 2379:2379 \
  -e ALLOW_NONE_AUTHENTICATION=yes \
  -v etcd_data:/etcd \
  bitnami/etcd:3.5.5

sleep 5

# 启动 MinIO
sudo docker run -d \
  --name milvus-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -v minio_data:/data \
  minio/minio:RELEASE.2023-03-20T20-16-18Z \
  server /data --console-address ":9001"

sleep 5

# 启动 Milvus
sudo docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -e ETCD_ENDPOINTS=etcd:2379 \
  -e MINIO_ADDRESS=minio:9000 \
  -v milvus_data:/var/lib/milvus \
  --link milvus-etcd:etcd \
  --link milvus-minio:minio \
  milvusdb/milvus:v2.3.4 \
  milvus run standalone

echo "⏳ 等待服务启动..."
sleep 30

echo "✅ 所有数据库启动完成!"
echo ""
echo "📊 访问信息:"
echo "   MongoDB: localhost:27017 (用户: root, 密码: root)"
echo "   Neo4j: http://localhost:7474 (用户: neo4j, 密码: 1q2w3e4r)"
echo "   Milvus: localhost:19530"
echo ""
echo "🔍 检查服务状态: docker ps"
