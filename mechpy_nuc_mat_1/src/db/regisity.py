from db.mongo import MongoClientManager
from db.neo4j import Neo4jClientManager
from db.milvus import MilvusClientManager

DataBase_CLIENT_MAP = {
    "neo4j": Neo4jClientManager,
    "mongo": MongoClientManager,
    "milvus": MilvusClientManager,
}