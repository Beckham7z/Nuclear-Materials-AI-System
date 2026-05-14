# -*- coding: utf-8 -*-
import yaml
from typing import Dict, Any , Optional
from dataclasses import dataclass, field,asdict, InitVar
from configuration.logset import logger
from llm.registry import INSTITUTION_CLIENT_MAP
from db.regisity import DataBase_CLIENT_MAP

@dataclass
class LLMConfig:
    model_type: str
    institution: str
    model: str
    prompt: str = ""
    user_message: Any = ""
    api_key: str = ""
    base_url: str = ""
    kwargs: Dict[str, Any] = field(default_factory=lambda: {}) # temperature': 0.7, 'max_tokens': 200
    client: Any = None

    def __post_init__(self):
        self._get_client()

    def _get_client(self):
        institution = self.institution.lower()
        client_cls = INSTITUTION_CLIENT_MAP.get(institution)
        if client_cls is not None:
            self.client = client_cls(self)
        else:
            self.client = None

    @classmethod
    def from_yaml_and_dict(cls, yaml_path: str, param_dict: Dict[str, Any] = None):
        # 1. 读取yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        llm_section = config.get("llm", {})
        # 2. 从param_dict获取model_type
        model_type = param_dict.get("model_type", "chat") if param_dict else "chat"
        default_institutions = {
            "chat": "zhipu",
            "embedding": "ollama",
            "vl": "zhipu"
        }
        institution = default_institutions.get(model_type.lower())
        llm_conf = llm_section.get(institution, {})

        # 3. 组装默认参数
        if model_type.lower() == "embedding":
            model = llm_conf.get("embed_model", "")
        elif model_type.lower() == "vl":
            model = llm_conf.get("vl_model", "")
        else:
            model = llm_conf.get("model", "")

        base_config = {
            "model_type": model_type,
            "institution": institution,
            "model": model,
            "prompt": "",
            "user_message": "",
            # "kwargs": {'temperature': 0.7, 'max_tokens': 200},
            "api_key": llm_conf.get("api_key", ""),
            "base_url": llm_conf.get("base_url", ""),
            "client": None  # 初始化时不创建客户端
        }

        # 4. 用param_dict覆盖
        if param_dict:
            base_config.update(param_dict)

        return cls(**base_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "institution": self.institution,
            "model": self.model,
            "prompt": self.prompt,
            "user_message": self.user_message,
            "kwargs": self.kwargs,
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
    

@dataclass
class Neo4jConfig:
    """Neo4j数据库配置"""
    url: str
    username: str
    password: str
    database: str
    batch_size_nodes: int = 500
    batch_size_edges: int = 100
    client: Any = None  # 用于存储Neo4j客户端实例

    def __post_init__(self):
        try:
            self.client = DataBase_CLIENT_MAP["neo4j"](self)
            logger.info("Neo4j客户端初始化成功")
        except Exception as e:
            logger.error(f"Neo4j客户端初始化失败: {str(e)}")
            self.client = None

@dataclass
class MongoConfig:
    """MongoDB数据库配置"""
    url: str
    database: str
    client: Any = None  # 用于存储MongoDB客户端实例

    def __post_init__(self):
        try:
            self.client = DataBase_CLIENT_MAP["mongo"](self)
            logger.info("Mongo客户端初始化成功")
        except Exception as e:
            logger.error(f"Mongo客户端初始化失败: {str(e)}")
            self.client = None

@dataclass
class MilvusConfig:
    """Milvus向量数据库配置"""
    url: str
    username: str
    password: str
    database: str
    client: Any = None  # 用于存储Milvus客户端实例

    def __post_init__(self):
        try:
            self.client = DataBase_CLIENT_MAP["milvus"](self)
            logger.info("Milvus客户端初始化成功")
        except Exception as e:
            logger.error(f"Milvus客户端初始化失败: {str(e)}")
            self.client = None

@dataclass
class DatabaseConfig:
    """数据库配置主类，包含所有数据库的配置信息"""
    neo4j: Neo4jConfig
    mongo: MongoConfig
    milvus: MilvusConfig

    @classmethod
    def from_dict(cls, config: dict) -> "DatabaseConfig":
        """从字典创建配置实例"""
        return cls(
            neo4j=Neo4jConfig(**config.get("neo4j", {})),
            mongo=MongoConfig(**config.get("mongo", {})),
            milvus=MilvusConfig(**config.get("milvus", {}))
        )
    


@dataclass()
class RAGConfig:
    Query_method: str = ""
    Question: str = ""
    top_k: int = 5
    concurrency : int = 5
    Q2K: bool = False
    use_Rerank:bool = False
    # ↓↓↓ 新增AI知识补全配置 ↓↓↓
    auto_complete: bool = True          # 是否允许AI补全缺失数据
    cite_source: bool = True            # 是否强制输出【检索】【AI补全】标签


    def to_dict(self) -> Dict[str, Any]:
        return {
            "Query_method": self.Query_method,
            "Question": self.Question,
            "Q2K": self.Q2K,
            "top_k": self.top_k,
            "concurrency": self.concurrency,
            "use_Rerank": self.use_Rerank,
            "auto_complete": self.auto_complete,
            "cite_source": self.cite_source
        }
    
@dataclass()
class FEMConfig:
    Solver: str = "chfem"
    analysis_type: str = "static"
    material_properties: Dict[str, Any] = field(default_factory=lambda: {
        "elastic_0_Young's modulus": 3.5,  # GPa
        "elastic_0_Poisson's ratio": 0.3,
        "elastic_255_Young's modulus": 0.01,  # GPa
        "elastic_255_Poisson's ratio": 0.3
    })
    mesh_size: float = 0.1  # mm
    boundary_conditions: Dict[str, Any] = field(default_factory=lambda: {}) # 边界条件 
    load_conditions: Dict[str, Any] = field(default_factory=lambda: {}) # 载荷条件
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Solver": self.Solver,
            "analysis_type": self.analysis_type,
            "material_properties": self.material_properties,
            "mesh_size": self.mesh_size,
            "boundary_conditions": self.boundary_conditions,
            "load_conditions": self.load_conditions
        }
