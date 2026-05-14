#!/bin/bash

echo "📥 手动安装数据库..."

# 安装 MongoDB
echo "🔧 安装 MongoDB..."
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt update
sudo apt install -y mongodb-org

sudo systemctl start mongod
sudo systemctl enable mongod
echo "✅ MongoDB 安装完成"

# 安装 Neo4j
echo "🔧 安装 Neo4j..."
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install -y neo4j

sudo systemctl start neo4j
sudo systemctl enable neo4j
sudo neo4j-admin set-initial-password 1q2w3e4r
echo "✅ Neo4j 安装完成"

# 安装 Milvus (使用 standalone 版本)
echo "🔧 安装 Milvus..."
# 下载 Milvus standalone
wget https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 如果有 docker-compose 就使用，否则手动安装组件
if command -v docker-compose &> /dev/null; then
    sudo docker-compose up -d
else
    echo "⚠️  需要 docker-compose 来安装 Milvus"
    echo "💡 请先解决网络问题后使用 Docker 安装"
fi

echo "✅ 手动安装完成"
