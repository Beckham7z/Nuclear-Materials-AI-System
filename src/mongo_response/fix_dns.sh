#!/bin/bash

echo "🔧 修复 DNS 配置..."

# 备份原始配置
sudo cp /etc/resolv.conf /etc/resolv.conf.backup

# 配置可靠的 DNS 服务器
sudo tee /etc/resolv.conf << 'RESOLV'
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 114.114.114.114
nameserver 223.5.5.5
search localdomain
options timeout:2 attempts:3
RESOLV

# 防止网络管理器覆盖配置
sudo chattr +i /etc/resolv.conf 2>/dev/null || true

echo "✅ DNS 配置完成"

# 测试 DNS 解析
echo "🔍 测试 DNS 解析..."
for domain in "registry-1.docker.io" "docker.mirrors.ustc.edu.cn" "github.com"; do
    echo -n "   $domain: "
    if nslookup $domain &> /dev/null; then
        echo "✅ 解析成功"
    else
        echo "❌ 解析失败"
    fi
done

# 重启 Docker 服务
echo "🔄 重启 Docker 服务..."
sudo systemctl restart docker

sleep 3

echo "🚀 现在尝试拉取数据库镜像..."
