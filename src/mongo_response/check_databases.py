import os
import sys
from pymongo import MongoClient
from neo4j import GraphDatabase
from pymilvus import connections, utility

def check_mongodb():
    """检查MongoDB数据库内容"""
    print("=" * 60)
    print("🔍 检查 MongoDB 数据库")
    print("=" * 60)
    
    try:
        # 连接MongoDB
        client = MongoClient("mongodb://root:root@localhost:27017/")
        
        # 检查连接状态
        print(f"✅ MongoDB 连接成功")
        print(f"   服务器版本: {client.server_info()['version']}")
        
        # 列出所有数据库
        databases = client.list_database_names()
        print(f"   可用数据库: {databases}")
        
        # 检查mech数据库
        if "mech" in databases:
            db = client["mech"]
            collections = db.list_collection_names()
            print(f"   mech数据库中的集合: {collections}")
            
            # 显示每个集合的文档数量
            for collection_name in collections:
                collection = db[collection_name]
                count = collection.count_documents({})
                print(f"     - {collection_name}: {count} 个文档")
                
                # 显示前几个文档的示例
                if count > 0:
                    sample_doc = collection.find_one()
                    print(f"       示例文档键: {list(sample_doc.keys())}")
        else:
            print("   ❌ mech数据库不存在")
            
        client.close()
        
    except Exception as e:
        print(f"❌ MongoDB 连接失败: {e}")

def check_neo4j():
    """检查Neo4j数据库内容"""
    print("\n" + "=" * 60)
    print("🔍 检查 Neo4j 数据库")
    print("=" * 60)
    
    try:
        # 连接Neo4j
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "1q2w3e4r")
        )
        
        # 测试连接
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            print(f"✅ Neo4j 连接成功")
            
            # 获取数据库信息
            result = session.run("CALL db.info()")
            db_info = result.single()
            print(f"   数据库名称: {db_info.get('name', 'N/A')}")
            print(f"   存储位置: {db_info.get('store', 'N/A')}")
            
            # 统计节点和关系
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"   节点总数: {node_count}")
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as relationship_count")
            relationship_count = result.single()["relationship_count"]
            print(f"   关系总数: {relationship_count}")
            
            # 显示节点标签
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            print(f"   节点标签: {labels}")
            
            # 显示关系类型
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            print(f"   关系类型: {rel_types}")
            
            # 显示一些示例节点
            if node_count > 0:
                result = session.run("MATCH (n) RETURN labels(n) as labels, properties(n) as props LIMIT 3")
                print("   示例节点:")
                for record in result:
                    print(f"     - 标签: {record['labels']}, 属性: {list(record['props'].keys())}")
                    
        driver.close()
        
    except Exception as e:
        print(f"❌ Neo4j 连接失败: {e}")

def check_milvus():
    """检查Milvus数据库内容"""
    print("\n" + "=" * 60)
    print("🔍 检查 Milvus 数据库")
    print("=" * 60)
    
    try:
        # 连接Milvus
        connections.connect(
            "default",
            uri="http://localhost:19530",
            user="root",
            password="root"
        )
        
        print(f"✅ Milvus 连接成功")
        
        # 列出所有集合
        collections = utility.list_collections()
        print(f"   可用集合: {collections}")
        
        # 检查mech数据库
        if "mech" in utility.list_databases():
            print(f"   mech数据库存在")
            
            # 切换到mech数据库
            connections.disconnect("default")
            connections.connect(
                "default",
                uri="http://localhost:19530",
                user="root",
                password="root",
                db_name="mech"
            )
            
            # 重新获取集合列表
            collections = utility.list_collections()
            print(f"   mech数据库中的集合: {collections}")
            
            # 显示每个集合的信息
            for collection_name in collections:
                if utility.has_collection(collection_name):
                    collection_info = utility.describe_collection(collection_name)
                    print(f"     - {collection_name}:")
                    print(f"       实体数量: {collection_info.get('entity_count', 'N/A')}")
                    print(f"       维度: {collection_info.get('dimension', 'N/A')}")
                    
        else:
            print("   ❌ mech数据库不存在")
            
        connections.disconnect("default")
        
    except Exception as e:
        print(f"❌ Milvus 连接失败: {e}")

def check_knowledge_base():
    """检查本地知识库文件"""
    print("\n" + "=" * 60)
    print("🔍 检查 本地知识库文件")
    print("=" * 60)
    
    kb_path = "/home/beckham7/A_project/n_material_file/knowledge_base/knowledge_base.json"
    
    try:
        import json
        
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
            
        print(f"✅ 知识库文件读取成功")
        print(f"   源文件: {kb_data.get('source_file', 'N/A')}")
        print(f"   创建时间: {kb_data.get('created_time', 'N/A')}")
        print(f"   文档块数量: {kb_data.get('total_chunks', 'N/A')}")
        
        chunks = kb_data.get('chunks', [])
        if chunks:
            print(f"   文档块示例:")
            for i, chunk in enumerate(chunks[:2]):  # 显示前2个块
                content_preview = chunk.get('content', '')[:100] + "..." if len(chunk.get('content', '')) > 100 else chunk.get('content', '')
                print(f"     - 块 {i+1}: {content_preview}")
                
    except Exception as e:
        print(f"❌ 知识库文件读取失败: {e}")

def main():
    """主函数"""
    print("🧠 数据库状态检查工具")
    print("=" * 60)
    
    # 检查各个数据库
    check_mongodb()
    check_neo4j()
    check_milvus()
    check_knowledge_base()
    
    print("\n" + "=" * 60)
    print("📊 检查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
