#!/bin/bash

set -e

echo "🚀 使用国内镜像源安装数据库..."

# 定义国内镜像源
MIRROR_USTC="docker.mirrors.ustc.edu.cn"
MIRROR_ALIYUN="registry.cn-hangzhou.aliyuncs.com"

# 拉取镜像函数
pull_image() {
    local image=$1
    local retries=3
    
    echo "📥 尝试拉取: $image"
    
    # 先尝试直接拉取
    if sudo docker pull $image; then
        echo "✅ 直接拉取成功: $image"
        return 0
    fi
    
    # 如果失败，尝试国内镜像源
    for mirror in $MIRROR_USTC $MIRROR_ALIYUN; do
        echo "🔄 尝试镜像源: $mirror"
        local mirror_image=""
        
        if [[ $image == *"/"* ]]; then
            # 处理有命名空间的镜像，如 milvusdb/milvus
            mirror_image="$mirror/$image"
        else
            # 处理官方镜像，如 mongo:latest -> library/mongo:latest
            mirror_image="$mirror/library/$image"
        fi
        
        if sudo docker pull $mirror_image; then
            echo "✅ 从 $mirror 拉取成功"
            # 重命名为原始镜像名
            sudo docker tag $mirror_image $image
            sudo docker rmi $mirror_image  # 删除镜像源标签
            return 0
        fi
    done
    
    echo "❌ 所有镜像源都失败: $image"
    return 1
}

echo "🧹 清理已存在的容器..."
sudo docker stop mongodb neo4j milvus-standalone milvus-etcd milvus-minio 2>/dev/null || true
sudo docker rm mongodb neo4j milvus-standalone milvus-etcd milvus-minio 2>/dev/null || true

echo "📥 开始拉取镜像..."
images=(
    "mongo:latest"
    "neo4j:latest" 
    "milvusdb/milvus:v2.3.4"
    "minio/minio:RELEASE.2023-03-20T20-16-18Z"
    "bitnami/etcd:3.5.5"
)

for image in "${images[@]}"; do
    if ! pull_image "$image"; then
        echo "⚠️  跳过 $image，继续安装其他组件"
    fi
done

echo "🚀 启动数据库服务..."

# 启动 MongoDB
if sudo docker images | grep -q "mongo"; then
    echo "启动 MongoDB..."
    sudo docker run -d \
      --name mongodb \
      -p 27017:27017 \
      -e MONGO_INITDB_ROOT_USERNAME=root \
      -e MONGO_INITDB_ROOT_PASSWORD=root \
      -v mongodb_data:/data/db \
      mongo:latest
else
    echo "❌ MongoDB 镜像不存在，跳过启动"
fi

# 启动 Neo4j
if sudo docker images | grep -q "neo4j"; then
    echo "启动 Neo4j..."
    sudo docker run -d \
      --name neo4j \
      -p 7474:7474 \
      -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/1q2w3e4r \
      -v neo4j_data:/data \
      -v neo4j_logs:/logs \
      -v neo4j_import:/var/lib/neo4j/import \
      neo4j:latest
else
    echo "❌ Neo4j 镜像不存在，跳过启动"
fi

# 启动 Milvus 组件
if sudo docker images | grep -q "bitnami/etcd"; then
    echo "启动 etcd..."
    sudo docker run -d \
      --name milvus-etcd \
      -p 2379:2379 \
      -e ALLOW_NONE_AUTHENTICATION=yes \
      -v etcd_data:/bitnami/etcd \
      bitnami/etcd:3.5.5
else
    echo "❌ etcd 镜像不存在"
fi

if sudo docker images | grep -q "minio/minio"; then
    echo "启动 MinIO..."
    sudo docker run -d \
      --name milvus-minio \
      -p 9000:9000 \
      -p 9001:9001 \
      -e MINIO_ROOT_USER=minioadmin \
      -e MINIO_ROOT_PASSWORD=minioadmin \
      -v minio_data:/data \
      minio/minio:RELEASE.2023-03-20T20-16-18Z \
      server /data --console-address ":9001"
else
    echo "❌ MinIO 镜像不存在"
fi

if sudo docker images | grep -q "milvusdb/milvus" && sudo docker ps | grep -q "milvus-etcd" && sudo docker ps | grep -q "milvus-minio"; then
    echo "启动 Milvus..."
    sudo docker run -d \
      --name milvus-standalone \
      -p 19530:19530 \
      -p 9091:9091 \
      -e ETCD_ENDPOINTS=milvus-etcd:2379 \
      -e MINIO_ADDRESS=milvus-minio:9000 \
      -v milvus_data:/var/lib/milvus \
      --link milvus-etcd:milvus-etcd \
      --link milvus-minio:milvus-minio \
      milvusdb/milvus:v2.3.4 \
      milvus run standalone
else
    echo "❌ Milvus 依赖不完整，跳过启动"
fi

echo "⏳ 等待服务启动..."
sleep 20

echo "✅ 安装完成!"
