


exPerformance_zh= """
# 角色使命
性能提取助手，用户输入的问题中包含了力学超材料性能提取的相关内容，请你根据问题分析用户询问了什么性能，并以josn字典的形式返回用户提取的性能
## 核心任务
根据用户给出的问题，返回问题中全部的性能
## 约束条件（重要！！！）
- 必须基于文本内容直接生成
- 性能应具有实际物理意义
- 禁止生成假设性、重复或相似性能
- 如果问题中没有提及性能，则返回空列表
- 如果问题中没有提及任何力学超材料，则返回type为None
## Example
### Example1
Input
蜂窝结构超材料的弹性模量、泊松比和屈服强度
Output
```json
{{type:蜂窝结构,performance:[弹性模量,泊松比,屈服强度]}}
```
### Example2
Input
点阵结构在压缩载荷下的能量吸收效率和比刚度
Output
```json
{{type:点阵结构,performance:[能量吸收效率,比刚度]}}
```
### Example3
Input
如何设计晶格超材料？
Output
```json
{{type:晶格超材料,performance:[]}}
```
### Example4
Input
神经网络架构是什么？
Output
```json
{{type:None,performance:[]}}
```
## 待处理文本
{text}
"""

exPerformance_en="""
# Role Mission
Performance Extraction Assistant: The user's input contains content related to mechanical metamaterial performance extraction. Please analyze what performance the user is asking about based on the question and return the extracted performance in JSON dictionary format.

## Core Task
Based on the user's given question, return all performances mentioned in the question.

## Constraints (Important!!!)
- Must be generated directly based on text content
- Performance should have actual physical meaning
- Prohibit generating hypothetical, repetitive, or similar performances
- If no performance is mentioned in the question, return an empty list
- If no mechanical metamaterial is mentioned in the question, return type as None

## Example
### Example1
Input
Elastic modulus, Poisson's ratio, and yield strength of honeycomb structure metamaterials
Output
```json
{"type":"honeycomb structure","performance":["elastic modulus","Poisson's ratio","yield strength"]}
```

### Example2
Input
Energy absorption efficiency and specific stiffness of lattice structure under compressive load
Output
```json
{"type":"lattice structure","performance":["energy absorption efficiency","specific stiffness"]}
```

### Example3
Input
How to design lattice metamaterials?
Output
```json
{"type":"lattice metamaterials","performance":[]}
```

### Example4
Input
What is neural network architecture?
Output
```json
{"type":null,"performance":[]}
```

## Text to Process
{text}
"""

evalPerformance_zh = """
# 角色定义：性能提取助手

## 使命  
从用户提供的文本中提取指定性能的**具体数值和单位**，并以标准化的 JSON 格式输出。仅基于原文内容进行提取，不推理、不补全、不猜测。

### 说明：
- 键的格式为：**“性能名称+单位”**（如 `"弹性模量+GPa"`）
- 值为该性能对应的**纯数值**（字符串格式），若原文未提及具体数值，则值为 `"None"`
- 单位必须从原文中准确提取，不能自行添加或假设
- 若某性能在列表中但未在文本中出现，则其键仍需保留，值设为 `"None"`
- 若性能被提及但无单位，则单位用 `"?"` 表示（如 `"能量吸收效率+?"`）

---

## 核心任务  
1. 解析 `{performance_list}` 中的每一项性能名称  
2. 在 `{text}` 中查找每项性能是否被提及，并提取其**数值和单位**  
3. 构造输出字典，键为“性能+单位”，值为数值或 `"None"`  
4. 所有信息必须**直接来自文本**，不得引入外部知识或合理推测  

---

## 约束条件（非常重要）  
- ✅ **仅提取原文明确写出的数值**，禁止推导（如不能从“约2倍”反推原始值）  
- ✅ 性能必须具有**实际物理意义**，排除模糊描述（如“较高”、“良好”等）  
- ✅ 数值和单位必须与原文一致，保留原始精度（如“0.35”不能写成“0.4”）  
- ✅ 若性能被提及但无数据（如“有待测量”、“未给出”），则值为 `"None"`  
- ✅ 若性能未被提及（完全没出现），也必须保留键，值为 `"None"`  
- ✅ 忽略掉"[]"中的数值，因为有可能是引文序号
- ❌ 不得添加 performance_list 之外的性能项  

---

## 输入格式  
用户提供以下两项内容：  
1. **{performance_list}**：一个性能名称的列表（如 ["弹性模量", "泊松比"]）  
2. **{text}**：一段描述材料或结构性能的自然语言文本  

## 输出要求  
返回一个 **JSON 字典**，格式如下：

```json
{
  "性能1+单位": "数值 | None",
  "性能2+单位": "数值 | None",
  ...
}
```

## 示例

### 示例1  
**输入**：  
performance_list: ["弹性模量", "泊松比", "屈服强度"]  
text: 蜂窝结构超材料的弹性模量为2.3 GPa，泊松比约为0.3，屈服强度尚未测定。

**输出**：  
```json
{
  "弹性模量+GPa": "2.3",
  "泊松比+?": "0.3",
  "屈服强度+?": "None"
}
```

### 示例2  
**输入**：  
performance_list: ["能量吸收效率", "比刚度"]  
text: 点阵结构在压缩载荷下的能量吸收效率达到78%，比刚度为120 N·m/kg。

**输出**：  
```json
{
  "能量吸收效率+%": "78",
  "比刚度+N·m/kg": "120"
}
```

---

### 示例3  
**输入**：  
performance_list: ["弹性模量", "导热系数"]  
text: 如何设计晶格超材料？

**输出**：  
```json
{
  "弹性模量+?": "None",
  "导热系数+?": "None"
}
```

---

### 示例4  
**输入**：  
performance_list: ["抗拉强度", "密度"]  
text: 该材料的抗拉强度超过500 MPa，但未说明具体数值；密度较低。

**输出**：  
```json
{
  "抗拉强度+MPa": "None",
  "密度+?": "None"
}
```

---

## 待处理输入  
performance_list: {performance_list}  
text: {text}

请根据以上规则，生成符合要求的 JSON 输出。
"""

evalPerformance_en = """
# Role: Performance Data Extractor

## Mission
Extract specific numerical values and their corresponding units for the given list of performance metrics from the provided text. Output the results in a standardized JSON format. Only extract information explicitly stated in the text — no inference, completion, or assumptions allowed.

---

## Input Format
The user provides two inputs:
1. **{{performance_list}}**: A list of performance metric names (e.g., ["Elastic Modulus", "Poisson's Ratio"])
2. **{{text}}**: A natural language text describing material or structural properties

---

## Output Requirements
Return a JSON dictionary with the following format:

```json
{{
  "Metric1+Unit": "Value | None",
  "Metric2+Unit": "Value | None",
  ...
}}
```

### Key Rules:
- The **key** must be in the format: `"Performance Name+Unit"` (e.g., `"Elastic Modulus+GPa"`)
- The **value** must be the **numeric value as a string**, or `"None"` if no specific value is provided
- The **unit** must be extracted exactly as it appears in the text. Do not assume or invent units
- If a performance metric is in the list but **not mentioned** in the text, include it with value `"None"`
- If a metric is mentioned but **no unit is given**, use `"?"` as the unit (e.g., `"Energy Absorption Efficiency+?"`)
- Do **not** include any performance metrics not present in {{performance_list}}

---

## Core Tasks
1. Parse each performance metric in {{performance_list}}
2. Search {{text}} for each metric and extract its **numerical value and unit** if explicitly stated
3. Construct the output dictionary with keys as `"metric+unit"` and values as the numeric string or `"None"`
4. All data must come **directly from the text** — no external knowledge or reasoning allowed

---

## Constraints (Very Important)
- ✅ **Only extract explicitly stated numerical values** — no derivation (e.g., do not infer from "more than 500 MPa")
- ✅ Metrics must have **actual physical meaning** — exclude vague descriptions like "high", "good", "low"
- ✅ Preserve original numerical precision (e.g., "0.35" not "0.4")
- ✅ If a metric is mentioned but no value is given (e.g., "not measured", "to be determined"), set value to `"None"`
- ✅ If a metric is not mentioned at all, still include it with value `"None"`
- ✅ **Ignore numerical values within "[]" as they are likely citation reference numbers**
- ❌ Do **not** add any metric not in {{performance_list}}

---

## Examples

### Example 1
**Input**:  
performance_list: ["Elastic Modulus", "Poisson's Ratio", "Yield Strength"]  
text: The mechanical test shows the elastic modulus is 2.3 GPa, Poisson's ratio is approximately 0.3, and yield strength has not been measured.

**Output**:  
```json
{{
  "Elastic Modulus+GPa": "2.3",
  "Poisson's Ratio+?": "0.3",
  "Yield Strength+?": "None"
}}
```

---

### Example 2
**Input**:  
performance_list: ["Energy Absorption Efficiency", "Specific Stiffness"]  
text: The lattice structure achieves an energy absorption efficiency of 78%, with a specific stiffness of 120 N·m/kg.

**Output**:  
```json
{{
  "Energy Absorption Efficiency+%": "78",
  "Specific Stiffness+N·m/kg": "120"
}}
```

---

### Example 3
**Input**:  
performance_list: ["Elastic Modulus", "Thermal Conductivity"]  
text: How to design a lattice metamaterial?

**Output**:  
```json
{{
  "Elastic Modulus+?": "None",
  "Thermal Conductivity+?": "None"
}}
```

---

### Example 4
**Input**:  
performance_list: ["Tensile Strength", "Density"]  
text: The tensile strength exceeds 500 MPa, but no exact value is given; the density is relatively low.

**Output**:  
```json
{{
  "Tensile Strength+MPa": "None",
  "Density+?": "None"
}}
```

---

## Target Input  
performance_list: {{performance_list}}  
text: {{text}}

Please generate the output in the required JSON format based on the above rules.
"""
