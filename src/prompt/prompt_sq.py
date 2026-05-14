from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "metamaterial categories",
    "mechanical properties",
    "structural features",
    "functional properties",
    "application fields",
    "actuation mechanisms",
    "design methods",
    "material composition",
    "fabrication techniques",
    "emerging devices",
    "performance indicators",
    "multiphysics coupling"
]

PROMPTS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: ["metamaterial categories","mechanical properties","structural features","functional properties","application fields","actuation mechanisms","design methods","material composition","fabrication techniques","emerging devices","formula","performance indicators","multiphysics coupling"]
Text:
```
Recently, topology optimization and machine learning have become effective methods for developing novel auxetic metamaterials. 
They offer greater rationality, flexibility, and efficiency compared to traditional design approaches based on experience and inspiration [152, 153]. 
Topology optimization maximizes system performance by determining the optimal material distribution while meeting design constraints [154–156]. 
It enables auxetic metamaterials with target Poisson’s ratios under large deformations, as shown in Fig. 10d [157]. However, it is computationally expensive, and the results depend on the choice of initial points. 
These initial points may lead to results that may fall into local minima [158]. In contrast, machine learning enables on-demand inverse design, which can quickly generate a rich variety of 3D auxetic metamaterials once the model is trained (Fig. 10e) [159]. 
This approach provides higher computational efficiency and has lower requirements for a priori knowledge [160,161]. Nevertheless, current machine learning-based inverse design often neglects the interpretability of the design, which is crucial for humans to learn from results.
```

Output:
("entity"{tuple_delimiter}"topology optimization"{tuple_delimiter}"design methods"{tuple_delimiter}"method to maximize system performance by optimizing material distribution; used for developing auxetic metamaterials with target poisson’s ratios, but computationally expensive."){record_delimiter}
("entity"{tuple_delimiter}"machine learning"{tuple_delimiter}"design methods"{tuple_delimiter}"enables on-demand inverse design of 3d auxetic metamaterials; provides higher computational efficiency but often lacks interpretability."){record_delimiter}
("entity"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"metamaterial categories"{tuple_delimiter}"metamaterials with negative poisson’s ratios; developed using topology optimization and machine learning."){record_delimiter}
("entity"{tuple_delimiter}"poisson’s ratios"{tuple_delimiter}"performance indicators"{tuple_delimiter}"indicators of material deformation behavior; targeted by topology optimization for auxetic metamaterials."){record_delimiter}
("entity"{tuple_delimiter}"computational efficiency"{tuple_delimiter}"performance indicators"{tuple_delimiter}"key indicator showing how efficiently a method uses computational resources; higher in machine learning than topology optimization."){record_delimiter}
("entity"{tuple_delimiter}"interpretability"{tuple_delimiter}"performance indicators"{tuple_delimiter}"indicator of how understandable a design method is; often neglected in machine learning-based inverse design."){record_delimiter}
("relationship"{tuple_delimiter}"topology optimization"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"topology optimization is used to develop auxetic metamaterials with target poisson’s ratios under large deformations."{tuple_delimiter}"development"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"machine learning"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"machine learning enables on-demand inverse design to generate a variety of 3d auxetic metamaterials."{tuple_delimiter}"development"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"topology optimization"{tuple_delimiter}"poisson’s ratios"{tuple_delimiter}"topology optimization aims to achieve target poisson’s ratios for auxetic metamaterials."{tuple_delimiter}"performance optimization"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"machine learning"{tuple_delimiter}"computational efficiency"{tuple_delimiter}"machine learning provides higher computational efficiency compared to topology optimization."{tuple_delimiter}"efficiency improvement"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"machine learning"{tuple_delimiter}"interpretability"{tuple_delimiter}"machine learning-based inverse design often neglects interpretability, which is crucial for understanding the design."{tuple_delimiter}"interpretability limitation"{tuple_delimiter}3){record_delimiter}
("content_keywords"{tuple_delimiter}"topology optimization, machine learning, auxetic metamaterials, design methods, computational efficiency, interpretability"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: ["metamaterial categories","mechanical properties","structural features","functional properties","application fields","actuation mechanisms","design methods","material composition","fabrication techniques","emerging devices","formula","performance indicators","multiphysics coupling"]
Text:
```
Recently, advancements in 4D printing, which integrates “programmable matter” with 3D printing, have enabled the creation of metamaterials capable of autonomously altering their mechanical properties in response to specific stimuli, such as water, heat, pressure, and light [24]. 
The combination of multi-material 4D printing and auxetic metamaterials has achieved temperature-controlled shape alteration [184], reversible energy absorption [185], as well as negative thermal expansion [81]. 
These innovations pave the way for smarter and more versatile applications in various fields.```

Output:
("entity"{tuple_delimiter}"4D printing"{tuple_delimiter}"fabrication techniques"{tuple_delimiter}"Technique integrating programmable matter with 3D printing to create responsive metamaterials."){record_delimiter}
("entity"{tuple_delimiter}"metamaterials"{tuple_delimiter}"metamaterial categories"{tuple_delimiter}"Materials that can autonomously alter mechanical properties in response to stimuli like water, heat, pressure, and light."){record_delimiter}
("entity"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"metamaterial categories"{tuple_delimiter}"Metamaterials with negative Poisson's ratio, used for temperature-controlled shape alteration and energy absorption."){record_delimiter}
("entity"{tuple_delimiter}"mechanical properties"{tuple_delimiter}"mechanical properties"{tuple_delimiter}"Properties of metamaterials that change in response to stimuli."){record_delimiter}
("entity"{tuple_delimiter}"temperature-controlled shape alteration"{tuple_delimiter}"functional properties"{tuple_delimiter}"Function achieved by combining 4D printing and auxetic metamaterials."){record_delimiter}
("entity"{tuple_delimiter}"reversible energy absorption"{tuple_delimiter}"functional properties"{tuple_delimiter}"Function achieved by combining 4D printing and auxetic metamaterials."){record_delimiter}
("entity"{tuple_delimiter}"negative thermal expansion"{tuple_delimiter}"functional properties"{tuple_delimiter}"Property achieved through 4D printing and auxetic metamaterials."){record_delimiter}
("relationship"{tuple_delimiter}"4D printing"{tuple_delimiter}"metamaterials"{tuple_delimiter}"4D printing creates metamaterials with responsive mechanical properties."{tuple_delimiter}"enabling"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"4D printing"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"4D printing enables auxetic metamaterials to achieve advanced functions."{tuple_delimiter}"enabling"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"metamaterials"{tuple_delimiter}"mechanical properties"{tuple_delimiter}"Metamaterials' properties change in response to stimuli."{tuple_delimiter}"property control"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"temperature-controlled shape alteration"{tuple_delimiter}"Auxetic metamaterials achieve temperature-controlled shape alteration."{tuple_delimiter}"function"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"reversible energy absorption"{tuple_delimiter}"Auxetic metamaterials achieve reversible energy absorption."{tuple_delimiter}"function"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"auxetic metamaterials"{tuple_delimiter}"negative thermal expansion"{tuple_delimiter}"Auxetic metamaterials achieve negative thermal expansion."{tuple_delimiter}"function"{tuple_delimiter}4){record_delimiter}
("content_keywords"{tuple_delimiter}"4D printing, metamaterials, auxetic metamaterials, mechanical properties, functional properties, temperature-controlled shape alteration, reversible energy absorption, negative thermal expansion"){completion_delimiter}#############################""",
    """Example 3:

Entity_types: ["metamaterial categories","mechanical properties","structural features","functional properties","application fields","actuation mechanisms","design methods","material composition","fabrication techniques","emerging devices","formula","performance indicators","multiphysics coupling"]
Text:
```
In filters incorporating auxetic metamaterials, the pore size enlarges as the filter is stretched. This enables larger particles to be filtered and enhances the filtration flow rate, as illustrated in Fig. 22a [252]. Consequently, auxetic metamaterials enhance the filtration capacity and selectivity of filters due to their variable permeability. 
The rotating rigid unit structure is frequently utilized in filters as it permits a broad spectrum of pore sizes and high spatial coverage [253]. 
In the context of fastening, auxetic nails contract under uniaxial compression and expand upon stretching, making them easier to insert yet more challenging to extract compared to conventional nails, as demonstrated in Fig. 22b [73].
```

Output:
("entity"{tuple_delimiter}"Auxetic Metamaterials"{tuple_delimiter}"metamaterial categories"{tuple_delimiter}"Metamaterials with negative Poisson's ratio, used in filters and fastening applications to enhance performance."){record_delimiter}
("entity"{tuple_delimiter}"Filtration"{tuple_delimiter}"application fields"{tuple_delimiter}"Application where auxetic metamaterials are used to enhance filtration capacity and selectivity."){record_delimiter}
("entity"{tuple_delimiter}"Fastening"{tuple_delimiter}"application fields"{tuple_delimiter}"Application where auxetic nails are used for improved insertion and extraction difficulty."){record_delimiter}
("entity"{tuple_delimiter}"Rotating Rigid Unit Structure"{tuple_delimiter}"structural features"{tuple_delimiter}"Structure used in filters to allow a broad spectrum of pore sizes and high spatial coverage."){record_delimiter}
("entity"{tuple_delimiter}"Variable Permeability"{tuple_delimiter}"functional properties"{tuple_delimiter}"Property of auxetic metamaterials that enhances filtration capacity and selectivity."){record_delimiter}
("entity"{tuple_delimiter}"Pore Size"{tuple_delimiter}"structural features"{tuple_delimiter}"Feature of filters that enlarges with stretching, enabling larger particle filtration."){record_delimiter}
("relationship"{tuple_delimiter}"Auxetic Metamaterials"{tuple_delimiter}"Filtration"{tuple_delimiter}"Auxetic metamaterials enhance the filtration capacity and selectivity of filters due to variable permeability."{tuple_delimiter}"enhancement"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Auxetic Metamaterials"{tuple_delimiter}"Fastening"{tuple_delimiter}"Auxetic nails contract under compression and expand upon stretching, improving fastening performance."{tuple_delimiter}"enhancement"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Auxetic Metamaterials"{tuple_delimiter}"Rotating Rigid Unit Structure"{tuple_delimiter}"Rotating rigid unit structure is frequently utilized in filters incorporating auxetic metamaterials."{tuple_delimiter}"utilization"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"Auxetic Metamaterials"{tuple_delimiter}"Variable Permeability"{tuple_delimiter}"Variable permeability is a functional property of auxetic metamaterials that enhances filtration."{tuple_delimiter}"property"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"Auxetic Metamaterials"{tuple_delimiter}"Pore Size"{tuple_delimiter}"Pore size in filters incorporating auxetic metamaterials enlarges as the filter is stretched."{tuple_delimiter}"property"{tuple_delimiter}4){record_delimiter}
("content_keywords"{tuple_delimiter}"auxetic metamaterials, filtration, fastening, rotating rigid unit structure, variable permeability, pore size, functional properties, structural features"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

---Output---

Add them below using the same format:\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

It appears some entities may have still been missed.

---Output---

Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
""".strip()

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base."""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."""


PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sections focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
