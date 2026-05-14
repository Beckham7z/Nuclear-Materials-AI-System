from pymongo import MongoClient
from configuration.logset import logger


class MongoClientManager:
    def __init__(self,mongoconfig):
        self.config = mongoconfig
        self.query_kword = {"collection_name": "text_chunks",  # "full_docts",
                            }
        self._create_client()

    def _create_client(self) -> MongoClient:
        self.client = MongoClient(self.config.url)[self.config.database]
        logger.debug(f"Mongo客户端初始化成功，连接到 {self.config.url} 数据库")
        # 输出都有哪些数据库
        # 输出数据库中的集合
        logger.debug(f"当前数据库({self.config.database})中的集合: {self.client.list_collection_names()}")
        # self.client.load_collection("fig_collection")
    
    def update(self, key: str, value: str):
        """
        更新MongoConfig中的指定键值对。
        只允许更新或添加在允许列表中的key，其它key不允许更新。
        """
        allowed_keys = ["collection_name", "data", "top_k", "search_params", "output_fields"]
        if key in allowed_keys:
            self.query_kword[key] = value
            logger.debug(f"MongoConfig更新成功: {key} = {value}")
        else:
            logger.debug(f"MongoConfig中不允许更新键 '{key}'。")

    def mongo_param_query(self, param_dict:dict) -> str:
        """
        根据_id字段检索指定集合中的文档，只返回content字段内容。
        :param collection_name: 集合名
        :param doc_id: 文档的_id
        :return: content字段内容（若未找到则返回None）
        """
        collection = self.query_kword["collection_name"]
        doc_list = self.client[collection].find(param_dict)
        if doc_list:
            logger.debug(f"集合 {collection} 中 {param_dict} 的content已检索到")
            return doc_list
        else:
            logger.debug(f"集合 {collection} 中未找到 {param_dict}")
            return None
        
    def mongo_id_query(self, doc_id: str) -> str:
        """
        根据_id字段检索指定集合中的文档，只返回content字段内容。
        :param collection_name: 集合名
        :param doc_id: 文档的_id
        :return: content字段内容（若未找到则返回None）
        """
        collection = self.query_kword["collection_name"]
        doc = self.client[collection].find_one({"_id": doc_id})
        if doc:
            logger.debug(f"集合 {collection} 中_id={doc_id} 的content已检索到")
            return doc
        else:
            logger.debug(f"集合 {collection} 中未找到 _id={doc_id}")
            return None
    
    def mongo_ids_query(self, doc_ids: list) -> list:
        """
        根据一组 _id 字段检索指定集合中的文档，返回所有匹配文档的内容。
        
        :param doc_ids: 文档 _id 的列表
        :return: 包含所有匹配文档的列表（每个元素为一个文档），未找到则返回空列表
        """
        collection = self.query_kword["collection_name"]
        cursor = self.client[collection].find({"_id": {"$in": doc_ids}})
        results = list(cursor)
        if results:
            logger.debug(f"集合 {collection} 中成功检索到 {len(results)} 个 _id 对应的文档")
        else:
            logger.debug(f"集合 {collection} 中未找到任何指定的 _id")

        return results