#!/bin/bash

echo "🔍 详细网络诊断..."
echo "=" * 60

echo "1. 基本网络检查:"
echo "   IP 地址: $(hostname -I)"
echo "   网关: $(ip route | grep default | awk '{print $3}')"

echo -e "\n2. DNS 解析测试:"
for domain in "registry-1.docker.io" "docker.mirrors.ustc.edu.cn" "github.com"; do
    echo -n "   $domain: "
    if nslookup $domain &> /dev/null; then
        echo "✅ 可解析"
    else
        echo "❌ 解析失败"
    fi
done

echo -e "\n3. 网络连通性测试:"
for target in "8.8.8.8" "93.179.102.140" "114.114.114.114"; do
    echo -n "   $target: "
    if ping -c 1 -W 1 $target &> /dev/null; then
        echo "✅ 可连接"
    else
        echo "❌ 连接失败"
    fi
done

echo -e "\n4. Docker 服务状态:"
sudo systemctl status docker --no-pager -l | grep -E "(Active|Main PID)"

echo -e "\n5. Docker 配置检查:"
sudo docker info 2>/dev/null | grep -E "(Registry Mirrors|HTTP Proxy|HTTPS Proxy)" || echo "   使用默认配置"

echo -e "\n6. 防火墙状态:"
sudo ufw status

echo -e "\n💡 根据以上诊断结果:"
echo "   - 如果 DNS 解析失败，需要配置 DNS"
echo "   - 如果网络不通，需要检查网络连接"
echo "   - 如果防火墙阻止，需要调整规则"
