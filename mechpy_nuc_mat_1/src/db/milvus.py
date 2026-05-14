from pymilvus import MilvusClient
from configuration.logset import logger



class MilvusClientManager:
    def __init__(self,milvusconfig):
        self.config = milvusconfig
        self.query_kword = {"collection_name": "entity", #"figure"
                            "data": None,
                            "top_k": 5, 
                            "search_params": {
                                "metric_type": "COSINE", 
                                "params": {}
                            },
                            "output_fields": ["id",]# "vector"
                            }
        self._create_client()

    def _create_client(self) -> MilvusClient:
        milvus_token = f"{self.config.username}:{self.config.password}"
        self.client = MilvusClient(uri=self.config.url, token=milvus_token)

        logger.debug(f"Milvus客户端初始化成功，连接到 {self.config.url} 数据库")

        # 输出都有哪些数据库
        logger.debug(f"所有数据库: {self.client.list_databases()}")

        self.client.use_database(db_name=self.config.database)
        # 输出数据库中的集合

        logger.debug(f"当前数据库({self.config.database})中的集合: {self.client.list_collections()}")
        # self.client.load_collection("fig_collection")

    
    
    def update(self, key: str, value: str):
        """
        更新MilvusConfig中的指定键值对。
        只允许更新或添加在允许列表中的key，其它key不允许更新。
        """
        allowed_keys = ["collection_name", "data", "top_k", "search_params", "output_fields"]
        if key in allowed_keys:
            self.query_kword[key] = value
            logger.debug(f"MilvusConfig更新成功: {key}")
        else:
            logger.debug(f"MilvusConfig中不允许更新键 '{key}'。")

    def milvus_query(self) -> list:
        # 从给定数据库中查询
        search_results = self.client.search(
            collection_name=self.query_kword["collection_name"],
            data=self.query_kword["data"],
            limit=self.query_kword["top_k"],
            search_params=self.query_kword["search_params"],
            output_fields=self.query_kword["output_fields"]
        )
        logger.debug(f"Milvus查询结果数量: {len(search_results)}")
        return search_results # 返回全部查询结果 这是一个列表
    




