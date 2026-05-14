#!/usr/bin/env python3
"""
RAG文档查询工具
用于查看已解析的文章列表和状态
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from datetime import datetime

def load_doc_status():
    """加载文档状态文件"""
    status_file = Path('myKG/kv_store_doc_status.json')
    if not status_file.exists():
        print("❌ 文档状态文件不存在")
        return {}
    
    with open(status_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_documents():

    # 获取当前时间并格式化
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    """分析文档状态"""
    doc_status = load_doc_status()
    
    if not doc_status:
        print("❌ 没有找到任何文档")
        return
    
    print("🚀 RAG 文档分析报告")
    print("=" * 50)
    
    # 统计不同状态的文档
    status_counts = {}
    processed_docs = []
    processing_docs = []
    pending_docs = []
    
    for doc_id, doc_info in doc_status.items():
        status = doc_info.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
        
        # 提取文档信息
        summary = doc_info.get('content_summary', '无摘要')
        chunks_count = doc_info.get('chunks_count', 0)
        created_at = doc_info.get('created_at', '未知时间')
        
        doc_info_display = {
            'id': doc_id,
            'status': status,
            'chunks': chunks_count,
            'summary': summary[:100] + '...' if len(summary) > 100 else summary,
            'created': created_at
        }
        
        if status == 'processed':
            processed_docs.append(doc_info_display)
        elif status == 'processing':
            processing_docs.append(doc_info_display)
        elif status == 'pending':
            pending_docs.append(doc_info_display)
    
    # 显示统计信息
    print(f"当前时间：{current_time}")
    print(f"📊 文档统计:")
    print(f"   ✅ 已处理完成: {status_counts.get('processed', 0)}")
    print(f"   🔄 处理中: {status_counts.get('processing', 0)}")
    print(f"   ⏳ 等待处理: {status_counts.get('pending', 0)}")
    print(f"   📄 总计: {len(doc_status)}")
    print()
    
    # # 显示已处理的文档
    # if processed_docs:
    #     print("✅ 已处理完成的文档:")
    #     print("-" * 40)
    #     for doc in processed_docs:
    #         print(f"📄 {doc['id'][:8]}...")
    #         print(f"   状态: {doc['status']}")
    #         print(f"   分块数: {doc['chunks']}")
    #         print(f"   摘要: {doc['summary']}")
    #         print(f"   创建时间: {doc['created']}")
    #         print()
    
    # # 显示处理中的文档
    # if processing_docs:
    #     print("🔄 处理中的文档:")
    #     print("-" * 40)
    #     for doc in processing_docs:
    #         print(f"📄 {doc['id'][:8]}...")
    #         print(f"   状态: {doc['status']}")
    #         print(f"   分块数: {doc['chunks']}")
    #         print(f"   摘要: {doc['summary']}")
    #         print()
    
    # # 显示等待处理的文档
    # if pending_docs:
    #     print("⏳ 等待处理的文档:")
    #     print("-" * 40)
    #     for doc in pending_docs[:5]:  # 只显示前5个
    #         print(f"📄 {doc['id'][:8]}...")
    #         print(f"   状态: {doc['status']}")
    #         print(f"   摘要: {doc['summary']}")
    #         print()
    #     if len(pending_docs) > 5:
    #         print(f"   ... 还有 {len(pending_docs) - 5} 个等待处理的文档")

def show_document_details(doc_id=None):
    """显示特定文档的详细信息"""
    doc_status = load_doc_status()
    
    if doc_id:
        if doc_id in doc_status:
            doc_info = doc_status[doc_id]
            print(f"📋 文档详情: {doc_id}")
            print("=" * 50)
            for key, value in doc_info.items():
                if key == 'content_summary':
                    print(f"{key}:")
                    print(f"  {value}")
                elif key == 'chunks_list' and value:
                    print(f"{key}: {len(value)} 个分块")
                    for i, chunk_id in enumerate(value[:5]):  # 只显示前5个
                        print(f"  - {chunk_id}")
                    if len(value) > 5:
                        print(f"  ... 还有 {len(value) - 5} 个分块")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"❌ 未找到文档: {doc_id}")
    else:
        # 如果没有指定文档ID，显示所有文档ID
        print("📄 所有文档ID:")
        print("-" * 30)
        for doc_id in doc_status.keys():
            status = doc_status[doc_id].get('status', 'unknown')
            print(f"{doc_id} [{status}]")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG文档查询工具')
    parser.add_argument('--details', '-d', help='显示特定文档的详细信息')
    parser.add_argument('--list', '-l', action='store_true', help='仅显示文档ID列表')
    
    args = parser.parse_args()
    
    if args.details:
        show_document_details(args.details)
    elif args.list:
        show_document_details()
    else:
        analyze_documents()

if __name__ == "__main__":
    main()
