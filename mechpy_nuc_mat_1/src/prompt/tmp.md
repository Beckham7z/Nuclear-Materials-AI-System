当然，以下是修改后的英文提示词版本，已将所有用于占位的 `{}` 替换为 `{{}}`，以兼容 Python 的 **f-string** 或 **`.format()`** 使用场景，避免格式化时出现 KeyError。

```python
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
```

---

✅ **使用说明**：
- 此字符串可安全用于 f-string 或 `.format()`：
  ```python
  prompt = evalPerformance_en.format(performance_list=["Elastic Modulus"], text="The elastic modulus is 1.5 GPa.")
  ```
- 所有占位符 `{{performance_list}}` 和 `{{text}}` 都会在运行时被正确替换
- JSON 块中的 `{{...}}` 是为了在最终输出中显示真实的大括号，符合 JSON 语法

如需进一步调整为 YAML、XML 或其他输出格式，也可继续扩展。