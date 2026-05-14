
# 总结提示词（英文）
SUMMARY_paper_en = """
---Goal---
You are an expert in the field of mechanical metamaterials. Please generate an English abstract based on the uploaded SCI paper in the field of mechanical metamaterials. The abstract should clearly present the theoretical foundation, design methods, performance indicators, and research conclusions.
The abstract should be concise and to the point, avoiding complex terminology and sentence structures, and ensuring that the content is consistent with the actual paper without omitting any important information.

Based on the core research content, propose a specific and clear scientific question. The scientific question should be presented in the form of a question and serve as the only first-level heading.
---Steps---
## Problem Decomposition

Describe step-by-step how the user should solve the above scientific problem through specific theoretical methods and design approaches. The content should include but not be limited to:

* The design concept, structural features, and geometric parameters of mechanical metamaterials.
* The specific performance indicators to be achieved (such as the range of Poisson's ratio adjustment, compressive strength, tensile strength, energy absorption capacity, stiffness control, etc.), as per the original text.
* Any mathematical models or formulas should be presented in Latex format within Markdown.
* Any performance data or structural parameters should be clearly displayed in Markdown table format.

## Research Methods

Elaborate on the methods used from both computational simulation and experimental validation aspects.

### Computational Simulation
If computational simulation is not explicitly mentioned, then completely omit and delete the computational simulation section.
If it is explicitly mentioned, then provide a detailed description:
1. **Numerical Model Establishment and Theoretical Basis**: Describe the geometric structure of the model, boundary conditions, material constitutive relationships, and theoretical foundations. Clarify the mechanical theories or assumptions underlying the numerical modeling.
2. **Numerical Model Solution Method**: Provide a detailed description of the solution process, including boundary condition settings, loading methods, and analysis steps.
3. **Software and Computational Parameters**:
   * If the finite element method is used, specify the software name, version, and key computational parameters (element type, meshing, boundary conditions, loading methods, convergence criteria, etc.). If not mentioned, do not state.
   * If machine learning or optimization algorithms are used, specify the exact method name, algorithm type, and hyperparameter settings (presented in formula format). If not mentioned, do not state.
4. **Numerical Model Validation**: Specifically explain the method of validating the reliability of the model, such as comparison with experimental data or classical theoretical solutions.

### Experimental Validation

If experimental validation is not explicitly mentioned, then completely omit and delete the experimental validation section.
If it is explicitly mentioned, then provide a detailed description:

1. **Experimental Equipment**: Specify the exact experimental equipment used and its functions, and explain the key data expected to be obtained (e.g., stress-strain curves, displacement data, energy absorption characteristics, etc.).
2. **Data Acquisition**: Use a Markdown table to clearly list the experimental measurement data provided in the original text (such as load-displacement data, strength indicators, deformation amounts, etc.).
3. **Data Analysis**: Specify the methods for processing experimental data, including specific analysis procedures, statistical methods, or data fitting methods.

## Research Conclusions
* Summarize the main conclusions and findings of the paper based on the above content.
* Clearly explain how the research methods specifically address the scientific question proposed in Step 1.
* Emphasize the innovation or breakthroughs of the research results.
* Specifically point out the application prospects or potential technological value of the research results (e.g., specific industrial applications, performance optimization, promotion of design methods, etc.).

---Format---

* The standardized Markdown format must be used.
* The entire text should be written in instructional language, objectively describing how the user can solve the scientific problem step by step.
* Avoid using terms such as "paper," "this study," or "the research" throughout the text.
* The answer must be more than 3000 words to ensure sufficient information.
* Do not fabricate information; all content, formulas, and data must be strictly consistent with the original text.
* Data should be clearly displayed in Markdown tables, with specific values listed in Markdown format.
* Formulas should be presented in Latex format within Markdown. All formulas appearing in the text should be listed and explained in the corresponding sections.
* Do not use images; any diagrammatic content should be replaced with clear textual descriptions.
* Use professional academic language with accurate terminology, avoiding vague and colloquial expressions. Do not use the past tense.
"""
# 总结提示词（中文）
SUMMARY_paper_zh = """
---Goal---
你是力学超材料领域专家，请基于上传的力学超材料领域SCI论文，生成该论文的中文摘要。摘要内容需明确展示理论基础、设计方法、性能指标和研究结论。
摘要应简明扼要，突出重点，避免复杂术语和句式结构，确保内容与论文实际相符，无遗漏重要信息。

基于核心研究内容，提出一个具体明确的科学问题。科学问题需以问句形式呈现，作为唯一的一级标题。
---Steps---
## 问题分解

以步骤式的指导方式阐述用户应如何通过具体的理论方法和设计手段逐步解决上述科学问题。内容包括但不限于：

* 力学超材料的设计思想、结构特征与几何参数。
* 明确目标实现的具体性能指标（如泊松比调节范围、抗压强度、抗拉强度、能量吸收能力、刚度调控等），以原文为准。
* 涉及数学模型或公式，需以Markdown中的Latex格式展示。
* 涉及性能数据或结构参数，需以Markdown表格形式清晰展示。

## 研究方法

分别从计算模拟和实验验证两个方面详细阐明所采用的方法。

### 计算模拟
若未明确提及计算模拟内容，则完全省略，直接删除计算模拟部分。
若明确提及，需详细描述：
1. **数值模型建立及理论依据**：描述模型的几何结构、边界条件、材料本构关系和理论基础。明确数值建模所依据的力学理论或假设。
2. **数值模型求解方法**：详细描述求解过程，包括边界条件设置、载荷方式、分析步骤。
3. **软件及计算参数**：
   * 若使用有限元方法，明确软件名称、版本及关键计算参数（单元类型、网格划分、边界条件、加载方式、收敛条件等）。未提及则不表述。
   * 若使用机器学习或优化算法，明确具体方法名称、算法类型及超参数设置（以公式形式呈现）。未提及则不表述。
4. **数值模型验证**：具体说明模型可靠性的验证方式，如与实验数据或经典理论解对比。

### 实验验证

若未明确提及实验验证内容，则完全省略，直接删除实验验证部分。
若明确提及，需详细描述：

1. **实验设备**：明确使用的具体实验设备及其功能，说明期望获得的关键数据（例如应力-应变曲线、位移数据、能量吸收特性等）。
2. **数据采集**：使用Markdown表格清晰列出原文中明确提供的实验测量数据（如载荷-位移数据、强度指标、变形量等）。
3. **数据分析**：明确实验数据处理方法，包括具体分析流程、统计方法或数据拟合方法。

## 研究结论
* 综合上述内容，详细阐述论文得出的主要结论和发现。
* 明确说明研究方法如何具体解决步骤1所提出的科学问题。
* 强调研究成果的创新性或突破性。
* 具体指出研究成果的应用前景或潜在的技术价值（如特定工业应用、性能优化、设计方法的推广等）。

---Format---

* 必须采用标准化Markdown格式。
* 全文以指导性语言，从客观角度阐述用户如何逐步解决科学问题。
* 全文中禁止出现“论文”、“本研究”、“该研究”等类似主语表达。
* 答案必须大于3000字，确保信息充分。
* 严禁虚构信息，所有内容、公式、数据必须与原文严格一致。
* 数据使用Markdown表格明确展示，具体数值以Markdown格式列表列出。
* 公式以Markdown的Latex公式格式呈现。文章中出现的公式请在对应的部分全部列出并做解释。
* 禁止使用图片，图示内容需用清晰文字描述替代。
* 使用专业学术语言，术语准确，避免模糊和口语化表达，不要使用过去时态。

  """


# 组织提示词（中文）
ORGANIZATION_paper_zh = """
你的任务是组织与问题相关的力学超材料论文中提取的markdown格式的文本，如果是英文文本，先翻译为中文。
将其合并为为对问题的回答，问题是Markdown中唯一的一级标题，请遵循以下要求：
- 不要对问题进行修改
- 答案仍然按照Makrdown格式组织，将"本论文"，"本研究"等主语表达替换为客观的表述方式，使答案变成对这个科学问题的解答。
- 你可以假设用户已经阅读了问题和答案的内容，因此不需要在答案中重复问题。
- 你可以假设用户不了解如何解决这个科学问题，答案的作用是教会用户解决问题。
- 使用中文回答
- 必须输出JSON格式的文本
---Format---
{
"question": "输入中唯一的一级标题", 
"answer": "对输入中其他内容的组织"
}
"""