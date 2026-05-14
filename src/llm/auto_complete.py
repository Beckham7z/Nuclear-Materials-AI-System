"""
AI知识补全模块
当检索结果技术密度不足时，自动补全关键数据并明确标注来源
"""

from typing import List, Tuple, Dict, Any
import re


def wrap_with_auto_complete(
    user_prompt: str,
    domain: str = "nuclear_materials"
) -> str:
    """
    万能包装器：任何用户提示词外面套一层"补全+溯源"指令，
    无需改原提示词，也保证回答不空谈。
    
    Args:
        user_prompt: 用户原始问题
        domain: 领域类型，默认为核材料
    
    Returns:
        包装后的提示词
    """
    return f"""你是核电材料领域的顶级专家，拥有完整的内部数据库和专业知识体系。

接下来你将收到：
1. 「用户原始问题」
2. 「检索到的相关文献片段」

请按照以下结构化流程进行分析：

## 分析要求

### 1. 技术密度评估
首先评估检索片段的技术密度：
- 如果检索片段包含**具体数值、实验数据、性能对比**，则优先基于检索内容
- 如果检索片段**技术密度不足**（缺数值、缺对比、缺成本、缺验证方法），则**必须**进行AI知识补全

### 2. 结构化输出格式
无论检索内容如何，请按以下结构输出：

#### 📊 关键指标对比表
- 包含至少3-4个候选材料
- 每个材料包含：关键性能指标（数值+单位）、优缺点、成本估算
- 使用表格格式，确保数据可读性

#### 🏆 推荐排名
- 基于综合评估给出1-2-3排名
- 每个推荐附带具体理由

#### 🔍 来源标注
**逐条标注信息来源**：
- 【检索】基于检索到的文献内容
- 【AI补全】基于专业知识补全的缺失数据

#### 📋 验证清单
对于【AI补全】部分，必须提供：
- 下一步实验验证建议
- 相关标准规范参考
- 潜在风险提示

### 3. 专业要求
- 使用专业术语但保持表述清晰
- 提供数值范围和不确定性说明
- 考虑实际工程应用的限制条件

---

## 用户原始问题
{user_prompt}

## 检索片段
{{retrieved_block}}

请开始你的专业分析："""


def extract_retrieved_content(mongo_results: List[Dict]) -> str:
    """
    从MongoDB检索结果中提取文本内容
    
    Args:
        mongo_results: MongoDB检索结果列表
        
    Returns:
        合并后的检索内容文本
    """
    if not mongo_results:
        return "本次检索未找到相关文献片段。"
    
    retrieved_texts = []
    for i, result in enumerate(mongo_results[:5]):  # 只取前5个结果
        text = result.get('text', '') or ''
        title = result.get('title', f"文档 {i+1}")
        score = result.get('score', 0)
        
        if text and len(text.strip()) > 10:  # 确保有实际内容
            retrieved_texts.append(f"""
【{title}】 (相似度: {score:.4f})
{text[:800]}{'...' if len(text) > 800 else ''}
            """.strip())
    
    if not retrieved_texts:
        return "检索到的文献内容为空或格式不支持。"
    
    return "\n\n".join(retrieved_texts)


def is_low_technical_density(text: str) -> bool:
    """
    判断文本技术密度是否过低
    
    Args:
        text: 待评估的文本
        
    Returns:
        True表示技术密度过低，需要补全
    """
    if not text or len(text.strip()) < 50:
        return True
    
    # 检查是否包含技术指标关键词
    technical_indicators = [
        'MPa', 'GPa', 'dpa', '℃', 'K', 'μm', 'mm', 'cm', 
        '腐蚀速率', '疲劳寿命', '蠕变', '肿胀', '强度', '韧性',
        '实验数据', '测试结果', '数值', '参数', '性能指标'
    ]
    
    technical_count = sum(1 for indicator in technical_indicators if indicator.lower() in text.lower())
    
    # 如果技术指标少于2个，认为技术密度过低
    return technical_count < 2


def format_auto_complete_response(response_text: str) -> str:
    """
    格式化AI补全的响应，增强可读性
    
    Args:
        response_text: 原始响应文本
        
    Returns:
        格式化后的响应文本
    """
    # 添加一些基本的格式化，实际使用时可以根据需要扩展
    formatted_response = response_text
    
    # 检测是否包含表格格式，如果没有则建议添加
    if '|' not in response_text and '表' not in response_text:
        formatted_response += "\n\n💡 **提示**: 建议在后续分析中使用表格格式展示关键指标对比，提高可读性。"
    
    return formatted_response
