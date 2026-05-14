# MongoDB集成使用指南 (更新版)

## 新的文件结构

### 主要文件位置
- **`zyx_test/mongo_utils.py`**: 所有MongoDB相关功能函数
- **`api/nuc_web_server.py`**: Web服务器，从zyx_test导入MongoDB功能
- **`md2mongo.py`**: MD到MongoDB处理器（主程序）
- **`test_md2mongo.py`**: 测试脚本

## 验证MongoDB中的数据

### 1. 验证数据存在性

```python
from zyx_test.mongo_utils import verify_mongo_data

# 验证MongoDB数据
verify_mongo_data()
```

### 2. 测试搜索功能

```python
from zyx_test.mongo_utils import search_mongo_documents

# 测试搜索
query = "CLAM steel irradiation"
results = search_mongo_documents(query, limit=5)

print(f'搜索 "{query}" 找到 {len(results)} 个文档')
for i, doc in enumerate(results):
    print(f'文档 {i+1}: {doc.get("header", "N/A")}')
```

## Web服务器中的MongoDB调用

### 1. 主要集成点

web服务器 (`api/nuc_web_server.py`) 现在从 `zyx_test/mongo_utils.py` 导入以下MongoDB功能：

- **`search_mongo_documents(query, limit)`**: 从MongoDB搜索相关文档
- **`build_enhanced_prompt(user_message, mongo_results)`**: 构建包含检索结果的增强提示词
- **`handle_mongo_query(global_config)`**: 独立的MongoDB查询处理函数

### 2. 在分析流程中的集成

在用户提交问题时，系统会自动：
1. 从MongoDB检索相关文档
2. 构建包含检索结果的增强提示词
3. 调用AI模型进行分析
4. 返回基于文档检索的分析结果

### 3. 独立MongoDB查询功能

```python
from zyx_test.mongo_utils import handle_mongo_query
from configuration.global_config import GlobalConfig

# 配置查询参数
global_config = GlobalConfig(
    rag=RAGConfig(
        Question="CLAM steel irradiation effects",
        top_k=10
    ),
    # ... 其他配置
)

# 执行MongoDB查询
result = handle_mongo_query(global_config)
```

## 数据验证结果

根据当前验证，您的MongoDB中已有：
- **总文档数**: 33个文档
- **主要文件**: `/home/beckham7/A_project/n_material_file/converted_output/converted.md` (32个文档)
- **测试文档**: 1个测试文档

## 启动和使用流程

### 1. 启动MongoDB服务

```bash
cd /home/beckham7/A_project/mechpy
./mongodb-linux-x86_64-ubuntu2404-8.2.1/bin/mongod --dbpath ./mongodb_data --bind_ip 0.0.0.0 --fork --logpath ./mongodb.log
```

### 2. 启动Web服务器

```bash
cd /home/beckham7/A_project/mechpy
conda activate N_RAG
streamlit run api/nuc_web_server.py
```

### 3. 测试查询

在web界面中输入以下测试问题：
- "CLAM steel irradiation effects"
- "nanocrystalline grains under Xe irradiation"
- "nanoindentation of irradiated materials"

## 新增功能说明

### 1. 智能文档检索
- 支持关键词搜索
- 自动去重和结果限制
- 宽松搜索策略（当精确匹配无结果时）

### 2. 增强提示词
- 自动整合检索结果到AI提示词
- 保持专业性和准确性
- 提供结构化的分析框架

### 3. 错误处理
- 数据库连接失败时的降级处理
- 搜索结果为空时的备用方案
- 详细的错误日志记录

## 性能优化建议

1. **索引优化**: 已自动创建文件路径、标题和时间戳索引
2. **搜索策略**: 支持正则表达式搜索，可扩展为向量搜索
3. **结果缓存**: 可添加结果缓存机制提高响应速度
4. **批量处理**: 支持批量文档处理

## 故障排除

### 常见问题

1. **连接失败**
   ```bash
   # 检查MongoDB服务状态
   ps aux | grep mongod
   # 重启服务
   pkill mongod
   ./mongodb-linux-x86_64-ubuntu2404-8.2.1/bin/mongod --dbpath ./mongodb_data --bind_ip 0.0.0.0 --fork --logpath ./mongodb.log
   ```

2. **搜索无结果**
   - 检查查询关键词是否准确
   - 验证文档内容是否包含相关关键词
   - 尝试更宽泛的搜索词

3. **内存不足**
   - 减少检索文档数量 (`top_k` 参数)
   - 优化分块大小

## 扩展功能

可根据需要扩展以下功能：
- 向量嵌入和相似性搜索
- 文档分类和标签系统
- 版本控制和增量更新
- 多语言文档支持

## 文件结构说明

```
/home/zyx/A_project/mechpy/
├── api/
│   └── nuc_web_server.py          # Web服务器（导入zyx_test功能）
├── zyx_test/
│   ├── mongo_utils.py             # MongoDB工具函数（新增）
│   └── MongoDB_Integration_Guide.md # 使用指南（新增）
├── md2mongo.py                    # MD到MongoDB处理器
├── test_md2mongo.py               # 测试脚本
└── mongodb-linux-x86_64-ubuntu2404-8.2.1/  # MongoDB安装包
```

## 测试功能

```python
# 测试所有功能
cd /home/beckham7/A_project/mechpy
conda activate N_RAG
python -c "from zyx_test.mongo_utils import verify_mongo_data, search_mongo_documents; verify_mongo_data(); print('搜索测试:', len(search_mongo_documents('CLAM steel', 3)))"
```

所有功能已成功迁移到zyx_test文件夹，web服务器路径已更新，系统可以正常工作！
