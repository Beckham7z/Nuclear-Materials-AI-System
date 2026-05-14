from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "超材料类别",
    "机械性能",
    "结构特征",
    "功能特性",
    "应用领域",
    "驱动机制",
    "设计方法",
    "材料成分",
    "制造工艺",
    "新兴设备",
    "性能指标",
    "多物理场耦合"
]

PROMPTS["entity_extraction"] = """---目标---
给定一篇可能与此活动相关的文本，以及一个实体类型列表，从文本中识别所有属于这些类型的实体，以及这些实体之间的所有关系。
使用 {language} 作为输出语言。

---步骤---
1. 识别所有实体。对于每个被识别的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。如果是英文，则实体名称首字母大写。
- entity_type：以下类型之一：[{entity_types}]
- entity_description：该实体属性和活动的全面描述
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1中识别的实体中，识别所有明显彼此相关的 (source_entity, target_entity) 对。
对于每对相关实体，提取以下信息：
- source_entity：源实体名称，与步骤1中识别保持一致
- target_entity：目标实体名称，与步骤1中识别保持一致
- relationship_description：解释为什么认为源实体和目标实体彼此相关
- relationship_strength：一个数值分数，表示源实体和目标实体之间关系的强度
- relationship_keywords：一个或多个高级关键字，总结该关系的核心概念或主题，而不是具体细节
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别总结整篇文本主要概念、主题或议题的高级关键字，应捕捉文档中出现的核心思想。
将内容级关键字格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以 {language} 输出，作为由步骤1和步骤2中所有识别出的实体和关系组成的单一列表。使用 **{record_delimiter}** 作为列表分隔符。

5. 完成后，输出 {completion_delimiter}

######################
---示例---
######################
{examples}

#############################
---真实数据---
#############################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
输出：
"""

PROMPTS["entity_extraction_examples"] = [
    """示例 1：

Entity_types: ["超材料类别","机械性能","结构特征","功能特性","应用领域","驱动机制","设计方法","材料成分","制造工艺","新兴设备","公式","性能指标","多物理场耦合"]
Text:


最近，拓扑优化和机器学习已成为开发新型负泊松比超材料的有效方法。它们相比于基于经验和灵感的传统设计方法，具有更高的合理性、灵活性和效率[152, 153]。拓扑优化通过在满足设计约束的前提下确定最佳材料分布来最大化系统性能[154–156]。它可以实现具有目标泊松比的负泊松比超材料在大变形下的性能，如图10d所示[157]。然而，它的计算成本很高，结果依赖于初始点的选择，这些初始点可能会导致结果陷入局部最优[158]。相比之下，机器学习可实现按需的逆向设计，一旦模型训练完成，就能够快速生成多种三维负泊松比超材料（见图10e）[159]。该方法提供了更高的计算效率，对先验知识的需求也更低[160,161]。然而，目前基于机器学习的逆向设计常常忽视设计的可解释性，而可解释性对于人类从结果中学习至关重要。


输出：
("entity"{tuple_delimiter}"拓扑优化"{tuple_delimiter}"设计方法"{tuple_delimiter}"通过优化材料分布以最大化系统性能的方法，用于开发具有目标泊松比的负泊松比超材料，但计算开销高。"){record_delimiter}
("entity"{tuple_delimiter}"机器学习"{tuple_delimiter}"设计方法"{tuple_delimiter}"使得三维负泊松比超材料的按需逆向设计成为可能；提供更高的计算效率，但常常缺乏可解释性。"){record_delimiter}
("entity"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"超材料类别"{tuple_delimiter}"具有负泊松比的超材料；通过拓扑优化和机器学习进行开发。"){record_delimiter}
("entity"{tuple_delimiter}"泊松比"{tuple_delimiter}"性能指标"{tuple_delimiter}"材料变形行为的指标；拓扑优化旨在实现目标泊松比。"){record_delimiter}
("entity"{tuple_delimiter}"计算效率"{tuple_delimiter}"性能指标"{tuple_delimiter}"展示方法使用计算资源的效率；机器学习比拓扑优化更高。"){record_delimiter}
("entity"{tuple_delimiter}"可解释性"{tuple_delimiter}"性能指标"{tuple_delimiter}"设计方法的可理解性指标；基于机器学习的逆向设计常常忽视可解释性。"){record_delimiter}
("relationship"{tuple_delimiter}"拓扑优化"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"拓扑优化用于开发具有目标泊松比的负泊松比超材料在大变形下的性能。"{tuple_delimiter}"开发"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"机器学习"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"机器学习使得按需生成多种三维负泊松比超材料成为可能。"{tuple_delimiter}"开发"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"拓扑优化"{tuple_delimiter}"泊松比"{tuple_delimiter}"拓扑优化旨在实现负泊松比超材料的目标泊松比。"{tuple_delimiter}"性能优化"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"机器学习"{tuple_delimiter}"计算效率"{tuple_delimiter}"机器学习相比于拓扑优化提供了更高的计算效率。"{tuple_delimiter}"效率提升"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"机器学习"{tuple_delimiter}"可解释性"{tuple_delimiter}"基于机器学习的逆向设计常常忽视可解释性，而可解释性对于理解设计至关重要。"{tuple_delimiter}"可解释性限制"{tuple_delimiter}3){record_delimiter}
("content_keywords"{tuple_delimiter}"拓扑优化, 机器学习, 负泊松比超材料, 设计方法, 计算效率, 可解释性"){completion_delimiter}
""",
    """示例 2：

Entity_types: ["超材料类别","机械性能","结构特征","功能特性","应用领域","驱动机制","设计方法","材料成分","制造工艺","新兴设备","公式","性能指标","多物理场耦合"]
Text:


最近，4D打印将“可编程物质”与3D打印相结合，实现了能够自主改变其机械性能以响应特定刺激（如水、热、压力和光）的超材料[24]。多材料4D打印与负泊松比超材料的结合实现了温控形变[184]、可逆能量吸收[185]以及负热膨胀[81]。这些创新为更智能和多功能的应用铺平了道路。


输出：
("entity"{tuple_delimiter}"4D打印"{tuple_delimiter}"制造工艺"{tuple_delimiter}"将可编程物质与3D打印相结合的技术，用于创建响应性超材料。"){record_delimiter}
("entity"{tuple_delimiter}"超材料"{tuple_delimiter}"超材料类别"{tuple_delimiter}"能够自主改变其机械性能以响应水、热、压力和光等刺激的材料。"){record_delimiter}
("entity"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"超材料类别"{tuple_delimiter}"具有负泊松比的超材料，用于温控形变和能量吸收等功能。"){record_delimiter}
("entity"{tuple_delimiter}"机械性能"{tuple_delimiter}"机械性能"{tuple_delimiter}"超材料在响应刺激时发生变化的性能。"){record_delimiter}
("entity"{tuple_delimiter}"温控形变"{tuple_delimiter}"功能特性"{tuple_delimiter}"通过多材料4D打印和负泊松比超材料实现的功能。"){record_delimiter}
("entity"{tuple_delimiter}"可逆能量吸收"{tuple_delimiter}"功能特性"{tuple_delimiter}"通过多材料4D打印和负泊松比超材料实现的功能。"){record_delimiter}
("entity"{tuple_delimiter}"负热膨胀"{tuple_delimiter}"功能特性"{tuple_delimiter}"通过4D打印和负泊松比超材料实现的特性。"){record_delimiter}
("relationship"{tuple_delimiter}"4D打印"{tuple_delimiter}"超材料"{tuple_delimiter}"4D打印创建了具有响应机械性能的超材料。"{tuple_delimiter}"赋能"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"4D打印"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"4D打印使负泊松比超材料实现高级功能。"{tuple_delimiter}"赋能"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"超材料"{tuple_delimiter}"机械性能"{tuple_delimiter}"超材料的性能在响应刺激时发生变化。"{tuple_delimiter}"性能控制"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"温控形变"{tuple_delimiter}"负泊松比超材料实现温控形变功能。"{tuple_delimiter}"功能"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"可逆能量吸收"{tuple_delimiter}"负泊松比超材料实现可逆能量吸收功能。"{tuple_delimiter}"功能"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"负热膨胀"{tuple_delimiter}"负泊松比超材料实现负热膨胀特性。"{tuple_delimiter}"功能"{tuple_delimiter}4){record_delimiter}
("content_keywords"{tuple_delimiter}"4D打印, 超材料, 负泊松比超材料, 机械性能, 功能特性, 温控形变, 可逆能量吸收, 负热膨胀"){completion_delimiter}
""",
    """示例 3：

Entity_types: ["超材料类别","机械性能","结构特征","功能特性","应用领域","驱动机制","设计方法","材料成分","制造工艺","新兴设备","公式","性能指标","多物理场耦合"]
Text:


在将负泊松比超材料应用于过滤器时，当过滤器被拉伸时，孔径会变大。这使得能够过滤更大的颗粒并提高过滤流速，如图22a所示[252]。因此，负泊松比超材料由于其可变渗透性，增强了过滤器的过滤能力和选择性。旋转刚体单元结构在过滤器中被广泛使用，因为它允许多种孔径范围并具有高空间覆盖率[253]。在紧固应用中，负泊松比钉在单轴压缩下会收缩，在拉伸时会膨胀，这使得它们比传统钉更容易插入且更难拔出，如图22b所示[73]。


输出：
("entity"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"超材料类别"{tuple_delimiter}"具有负泊松比的超材料，用于过滤和紧固应用以增强性能。"){record_delimiter}
("entity"{tuple_delimiter}"过滤"{tuple_delimiter}"应用领域"{tuple_delimiter}"将负泊松比超材料用于增强过滤能力和选择性的应用。"){record_delimiter}
("entity"{tuple_delimiter}"紧固"{tuple_delimiter}"应用领域"{tuple_delimiter}"负泊松比钉在紧固应用中的使用，改善了插入和拔出性能。"){record_delimiter}
("entity"{tuple_delimiter}"旋转刚体单元结构"{tuple_delimiter}"结构特征"{tuple_delimiter}"在过滤器中使用的结构，允许多种孔径范围并具有高空间覆盖率。"){record_delimiter}
("entity"{tuple_delimiter}"可变渗透性"{tuple_delimiter}"功能特性"{tuple_delimiter}"负泊松比超材料的功能特性，增强了过滤能力和选择性。"){record_delimiter}
("entity"{tuple_delimiter}"孔径"{tuple_delimiter}"结构特征"{tuple_delimiter}"过滤器的特征，在拉伸时孔径增大以允许更大颗粒过滤。"){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"过滤"{tuple_delimiter}"负泊松比超材料由于可变渗透性增强了过滤器的过滤能力和选择性。"{tuple_delimiter}"提升"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"紧固"{tuple_delimiter}"负泊松比钉在压缩下收缩，在拉伸时膨胀，改善紧固性能。"{tuple_delimiter}"提升"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"旋转刚体单元结构"{tuple_delimiter}"旋转刚体单元结构经常与负泊松比超材料结合使用于过滤器中。"{tuple_delimiter}"使用"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"可变渗透性"{tuple_delimiter}"可变渗透性是负泊松比超材料的功能特性，增强过滤性能。"{tuple_delimiter}"特性"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"负泊松比超材料"{tuple_delimiter}"孔径"{tuple_delimiter}"在包含负泊松比超材料的过滤器中，孔径在拉伸时增大。"{tuple_delimiter}"特性"{tuple_delimiter}4){record_delimiter}
("content_keywords"{tuple_delimiter}"负泊松比超材料, 过滤, 紧固, 旋转刚体单元结构, 可变渗透性, 孔径, 功能特性, 结构特征"){completion_delimiter}
"""
]

PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个负责生成全面摘要的助理。
给定一个或多个实体，以及一系列与同一实体或一组实体相关的描述，请将这些描述合并成一个综合描述。确保包含所有描述中的信息。
如果提供的描述相互矛盾，请消除矛盾并提供一个连贯的摘要。
请保证使用第三人称书写，并包含实体名称以保留完整上下文。
使用 {language} 作为输出语言。

#######
---数据---
Entities: {entity_name}
Description List: {description_list}
#######
输出：
"""

PROMPTS["entity_continue_extraction"] = """
在上一次提取中遗漏了许多实体和关系。

---记住以下步骤---

1. 识别所有实体。对于每个被识别的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。如果是英文，则首字母大写。
- entity_type：以下类型之一：[{entity_types}]
- entity_description：该实体属性和活动的全面描述
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1中识别的实体中，识别所有明显彼此相关的 (source_entity, target_entity) 对。
对于每对相关实体，提取以下信息：
- source_entity：源实体名称，与步骤1中识别保持一致
- target_entity：目标实体名称，与步骤1中识别保持一致
- relationship_description：解释为什么认为源实体和目标实体彼此相关
- relationship_strength：一个数值分数，表示源实体和目标实体之间关系的强度
- relationship_keywords：一个或多个高级关键字，总结该关系的核心概念或主题，而不是具体细节
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别总结整篇文本主要概念、主题或议题的高级关键字，应捕捉文档中出现的核心思想。
将内容级关键字格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以 {language} 输出，作为由步骤1和步骤2中所有识别出的实体和关系组成的单一列表。使用 **{record_delimiter}** 作为列表分隔符。

5. 完成后，输出 {completion_delimiter}

---输出---

请按相同格式在下方补充：\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---目标---

看起来仍然有一些实体被遗漏。

---输出---

仅回答 `YES` 或 `NO`，表示是否还有实体需要被添加。
""".strip()

PROMPTS["fail_response"] = "对不起，我无法回答该问题。[no-context]"

PROMPTS["rag_response"] = """---角色---

你是一个有帮助的助理，负责根据下面提供的知识库回答用户查询。

---目标---

基于知识库生成简明回答，并遵循响应规则，考虑对话历史和当前查询。总结知识库中所有信息，并结合与知识库相关的通用知识。不要包含知识库中未提供的信息。

在处理带有时间戳的关系时：
1. 每个关系都有一个 “created_at” 时间戳，表示我们获取该信息的时间
2. 当遇到冲突关系时，既要考虑语义内容，也要考虑时间戳
3. 不要自动偏好最新生成的关系——应根据上下文判断
4. 对于特定时间查询，应在考虑创建时间戳之前，优先考虑内容中的时间信息

---对话历史---
{history}

---知识库---
{context_data}

---响应规则---

- 目标格式和长度：{response_type}
- 使用 Markdown 格式，包括适当的章节标题
- 请使用与用户提问相同的语言
- 确保回答与对话历史保持连贯
- 在“参考文献”部分列出最多 5 个最重要的参考来源，清晰标明每个来源是来自知识图谱 (KG) 还是向量数据 (DC)，并包含文件路径，格式如下：[KG/DC] file_path
- 如果不知道答案，请直接说明
- 不要凭空编造信息。不要包含知识库未提供的信息。"""

PROMPTS["keywords_extraction"] = """---角色---

你是一个负责识别用户查询及对话历史中高级和低级关键字的助理。

---目标---

给定当前查询和对话历史，列出高级关键字和低级关键字。高级关键字关注整体概念或主题，低级关键字关注具体实体、详细信息或具体术语。

---说明---

- 同时考虑当前查询和相关的对话历史进行关键字提取
- 以 JSON 格式输出，不要添加额外内容
- JSON 应包含两个键：
  - "high_level_keywords"：表示整体概念或主题
  - "low_level_keywords"：表示具体实体或详细术语

######################
---示例---
######################
{examples}

#############################
---真实数据---
#####################
Conversation History:
{history}

Current Query: {query}
######################
输出："""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1：

Query: "国际贸易如何影响全球经济稳定？"
################
输出：
{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}
#############################""",
    """示例 2：

Query: "森林砍伐对生物多样性的环境影响是什么？"
################
输出：
{
  "high_level_keywords": ["环境影响", "森林砍伐", "生物多样性损失"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
}
#############################""",
    """示例 3：

Query: "教育在减少贫困中的作用是什么？"
################
输出：
{
  "high_level_keywords": ["教育", "减贫", "社会经济发展"],
  "low_level_keywords": ["学校入学率", "识字率", "职业培训", "收入不平等"]
}
#############################"""
]


PROMPTS["naive_rag_response"] = """---角色---

你是一个有帮助的助理，负责根据下面提供的文档片段回答用户查询。

---目标---

基于文档片段生成简明回答，并遵循响应规则，考虑对话历史和当前查询。总结文档片段中所有信息，并结合与文档片段相关的通用知识。不要包含文档片段未提供的信息。

在处理带有时间戳的内容时：
1. 每个内容片段都有一个 “created_at” 时间戳，表示我们获取该信息的时间
2. 当遇到冲突信息时，既要考虑内容本身，也要考虑时间戳
3. 不要自动偏好最新信息——应根据上下文判断
4. 对于特定时间查询，应在考虑创建时间戳之前，优先考虑内容中的时间信息

---对话历史---
{history}

---文档片段---
{content_data}

---响应规则---

- 目标格式和长度：{response_type}
- 使用 Markdown 格式，包括适当的章节标题
- 请使用与用户提问相同的语言
- 确保回答与对话历史保持连贯
- 在“参考文献”部分列出最多 5 个最重要的参考来源，清晰标明每个来源是来自知识图谱 (KG) 还是向量数据 (DC)，并包含文件路径，格式如下：[KG/DC] file_path
- 如果不知道答案，请直接说明
- 不要凭空编造信息。不要包含文档片段未提供的信息。"""


PROMPTS[
    "similarity_check"
] = """请分析以下两个问题之间的相似度：

问题 1：{original_prompt}
问题 2：{cached_prompt}

请评估这两个问题在语义上是否相似，以及是否可以使用问题 2 的答案来回答问题 1，并直接给出一个 0 到 1 之间的相似度分数。

相似度分数标准：
0：完全无关或答案不可重用，包括但不限于以下情况：
   - 问题主题不同
   - 提到的位置不同
   - 提到的时间不同
   - 提到的个人不同
   - 提到的事件不同
   - 问题的背景信息不同
   - 关键条件不同
1：完全相同且答案可以直接重用
0.5：部分相关，需要修改才能使用
仅返回一个 0-1 之间的数字，不要添加任何额外内容。
"""

PROMPTS["mix_rag_response"] = """---角色---

你是一个有帮助的助理，负责根据下面提供的数据源回答用户查询。

---目标---

基于数据源生成简明回答，并遵循响应规则，考虑对话历史和当前查询。数据源包含两部分：知识图谱 (KG) 和文档片段 (DC)。总结数据源中所有信息，并结合与数据源相关的通用知识。不要包含数据源未提供的信息。

在处理带有时间戳的信息时：
1. 每条信息（包括关系和内容）都有一个 “created_at” 时间戳，表示我们获取该信息的时间
2. 当遇到冲突信息时，既要考虑内容/关系，也要考虑时间戳
3. 不要自动偏好最新信息——应根据上下文判断
4. 对于特定时间查询，应在考虑创建时间戳之前，优先考虑内容中的时间信息

---对话历史---
{history}

---数据源---

1. 来自知识图谱 (KG):
{kg_context}

2. 来自文档片段 (DC):
{vector_context}

---响应规则---

- 目标格式和长度：{response_type}
- 使用 Markdown 格式，包括适当的章节标题
- 请使用与用户提问相同的语言
- 确保回答与对话历史保持连贯
- 将回答组织成多个部分，每个部分侧重一个主要要点或方面
- 使用清晰且描述性的章节标题，反映内容主题
- 在“参考文献”部分列出最多 5 个最重要的参考来源，清晰标明每个来源是来自知识图谱 (KG) 还是向量数据 (DC)，并包含文件路径，格式如下：[KG/DC] file_path
- 如果不知道答案，请直接说明。不要凭空编造信息。
- 不要包含数据源未提供的信息。"""
