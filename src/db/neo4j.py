import os
import shutil
from neo4j import GraphDatabase
from configuration.logset import logger


class Neo4jClientManager:
    def __init__(self, neo4jconfig):
        self.config = neo4jconfig
        self.query_kword = {"batch_size_nodes": None,
                             "batch_size_edges": None, 
                             "cypher": None,
                             "properties": {"Figure":["doi", "image_path", "fig_desc", "description","base64"],
                                            "DOI": ["doi","bibtex","title"],
                                            "Abstract":["doi","abstract"],
                                            "Entity":["entity_id", "entity_type", "description", "source_id"],
                                            },
                            }
        self._get_client()

    def _get_client(self) -> GraphDatabase.driver:
        """创建并返回Neo4j客户端"""
        try:
            self.client = GraphDatabase.driver(
                self.config.url,
                auth=(self.config.username, self.config.password)
            )
            logger.debug(f"Neo4j客户端初始化成功")

        except Exception as e:
            logger.error(f"Neo4j客户端初始化失败: {str(e)}")
            return None
        
    def update(self, key: str, value: str):
        """
        更新Neo4jConfig中的指定键值对。
        只允许更新或添加在允许列表中的key，其它key不允许更新。
        """
        allowed_keys = ["batch_size_nodes", "batch_size_edges", "cypher"]
        if key in allowed_keys:
            self.query_kword[key] = value
            logger.debug(f"Neo4jConfig更新成功: {key} = {value}")
        else:
            logger.debug(f"Neo4jConfig中不允许更新键 '{key}'。")

    def image_first_neo4j(self, doi:str, image_path: str) -> dict:
        query = """
        MATCH (a:Figure {doi: $doi, image_path: $image_path})
        RETURN a
        """
        with self.client.session() as session:
            result = session.run(query, doi=doi,image_path=image_path).data()
        # logger.debug(f"Neo4j查询结果: {result}")
        return {"image_node": result} if result else {}
    
    def old_image_first_neo4j(self, fig_desc:str) -> dict:
        query = """
        MATCH (n:Figure{fig_desc: $fig_desc})
        OPTIONAL MATCH (n)<-[r]-(d:DOI)
        RETURN n, d.id AS doi_id
        """
        with self.client.session() as session:
            result = session.run(query, fig_desc=fig_desc).data()
        return {"image_node": result} if result else {}

    
    def abstract_first_neo4j(self, doi: str) -> dict:
        """
        输入是检索增强生成得到的最近似摘要的DOI
        输出是该DOI对应的摘要、实体和图像 node dict
        """
        query = """
        MATCH (a:Abstract {doi: $doi})
        OPTIONAL MATCH (d:DOI {doi: $doi})
        OPTIONAL MATCH (d)-[:CONTAINS]->(e:Entity)
        OPTIONAL MATCH (d)-[:CONTAIN]->(f:Figure)
        RETURN a.abstract AS abstract, 
            collect(DISTINCT e) AS entities, 
            collect(DISTINCT f) AS figures
        """
        with self.client.session() as session:
            result = session.run(query, doi=doi).data()
            if result:
                abstract_str = result[0].get("abstract", "")
                entities_list = [node for node in result[0]['entities'] if node is not None]
                figures_list = [node for node in result[0]['figures'] if node is not None]
            else:
                abstract_str, entities_list, figures_list = "", [], []
        return {
            "abstract": abstract_str,
            "entities": entities_list,
            "figures": figures_list
        }
    
    def node_first_neo4j(self, entity_id: str) -> dict:
        """
        查询实体节点本体和其直接关联的实体节点，用于NodeQueryEngine构建上下文。
        """
        query = """
            MATCH (core:Entity {entity_id: $entity_id})
            OPTIONAL MATCH (core)-[:DIRECT*1..1]-(linked:Entity)
            RETURN core, collect(DISTINCT linked) AS entities
            """
        try:
            with self.client.session() as session:
                result = session.run(query, entity_id=entity_id).data()
                if result:
                    record = result[0]
                    core_entity = record.get("core")
                    entities = [e for e in record.get("entities", []) if e is not None]
                    return {
                        "core-entity": core_entity,
                        "entities": entities
                    }
                return {"core-entity": None, "entities": []}
        except Exception as e:
            logger.error(f"Neo4j 实体节点查询失败: {e}")
            return {"core-entity": None, "entities": []}
            
    def neo4j_query(self) -> dict:
        """执行Neo4j查询"""
        if not self.client:
            logger.error("Neo4j客户端未初始化，无法执行查询")
            return []

        try:
            with self.client.session() as session:
                result = session.run(self.query_kword["cypher"])
                return {"node":result.data()}  # 返回查询结果字典

        except Exception as e:
            logger.error(f"Neo4j查询失败: {str(e)}")
            return {"node":[]}
        
    def neo4j_parameterized_query(self, cypher: str, parameters: dict) -> dict:
        """
        执行带参数的 Neo4j 查询
        """
        if not self.client:
            logger.error("Neo4j客户端未初始化，无法执行查询")
            return {"node": []}

        try:
            with self.client.session() as session:
                logger.debug(f"执行的 Cypher 查询语句: {cypher}，参数: {parameters}")
                result = session.run(cypher, **parameters)
                return {"node": result.data()}
        except Exception as e:
            logger.error(f"Neo4j查询失败: {str(e)}")
            return {"node": []}
        
    def export2graphml(self, target_dir: str,import_dir: str = "/var/lib/neo4j/export",export_file: str = "export.graphml"): # 未测试
        """
        将 Neo4j import 目录下的导出文件移动到指定目录
        :param import_dir: Neo4j 的 import 目录
        :param export_file: 导出的文件名或相对路径
        :param target_dir: 目标目录（绝对路径或相对路径）
        """
        cypher = f'CALL apoc.export.graphml.all("{export_file}", {{}})'
        try:
            with self.client.session() as session:
                session.run(cypher)
                logger.info(f"GraphML 导出成功，文件路径: {os.path.join(import_dir,export_file)}")
        except Exception as e:
            logger.error(f"GraphML 导出失败: {str(e)}")
        src_path = os.path.join(import_dir, export_file)
        dst_path = os.path.join(target_dir, os.path.basename(export_file))
        try:
            shutil.move(src_path, dst_path)
            logger.info(f"文件已移动到: {dst_path}")
        except Exception as e:
           logger.error(f"移动文件失败: {e}")

    

query_fig = """
        MATCH (n:Figure{})
        WHERE id(n) = $node_id
        OPTIONAL MATCH (n)<-[r]-(d:DOI)
        RETURN n
        """