from __future__ import annotations
from typing import Any



GRAPH_FIELD_SEP = "<SEP>"

EX_kword_zh = """---Goal---
请从下述问题中提取出最核心的关键词，并对每个关键词进行重要性评分（1~5分，5分为最重要），关键词以如下JSON格式输出：
```json
{"keyword":{"关键词1": 评分1,"关键词2": 评分2,..."关键词N": 评分N}}
```
######################
---Examples---
######################
Example 1:
input:
如何设计负泊松比超材料？
output:
```json
{"keyword": {"负泊松比超材料": 5,"设计": 4}}
```
######################
Example 2:
input:
基于拓扑优化方法，如何设计具有高能量吸收能力的三维力学超材料结构？
output:
```json
{"keyword": {"拓扑优化": 5,"设计": 4,"高能量吸收": 5,"三维力学超材料": 5,"结构": 3}}
```
######################
Example 3:
input:
如何利用增材制造技术制备具有可编程力学性能的超材料，并实现其在柔性机器人中的应用？
output:
```json
{"keyword": {"增材制造": 5,"可编程力学性能超材料": 5,"柔性机器人": 4,"应用": 3}}
```
#############################
---Real Data---
######################
Text:
{input_question}
######################
Output:"""

EX_kword_en = """---Goal---
Please extract the most essential keywords from the following question and assign an importance score to each keyword (1~5, with 5 being the most important). Output the keywords in the following JSON format:
```json
{"keyword": {"keyword1": score1,"keyword2": score2,..."keywordN": scoreN}}
```
######################
---Examples---
######################
Example 1:
input:
How to design auxetic metamaterials?
output:
```json
{"keyword": {"auxetic metamaterials": 5,"design": 4}}
```
######################
Example 2:
input:
Based on topology optimization, how to design three-dimensional mechanical metamaterial structures with high energy absorption capacity?
output:
```json
{"keyword": {"topology optimization": 5,"design": 4,"high energy absorption": 5,
"three-dimensional mechanical metamaterials": 5,"structures": 3}}
```
######################
Example 3:
input:
How to use additive manufacturing technology to fabricate metamaterials with programmable mechanical properties and realize their application in soft robotics?
output:
```json
{"keyword": {"additive manufacturing": 5,"programmable mechanical properties metamaterials": 4,"soft robotics": 4,"application": 3}}
```
#############################
---Real Data---
######################
Text:
{input_question}
######################
Output:"""