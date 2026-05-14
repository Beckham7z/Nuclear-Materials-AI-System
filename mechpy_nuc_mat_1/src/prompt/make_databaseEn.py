# This file contains prompts for generating fine-tuning datasets in the field of mechanical metamaterials.

TXT2Q_prompt = """
# Role Mission
You are an expert in analyzing academic texts in the field of mechanical metamaterials, skilled at extracting key information from complex texts and generating structured data for model fine-tuning (generate questions only).

## Core Task
Based on the provided text, generate no fewer than {number} high-quality questions.

## Constraints (Important!!!)
- Must be generated directly based on the text content
- Questions should have clear answer directionality
- Cover different aspects of the text
- Do not generate hypothetical, duplicate, or similar questions

## Special Requirements - Genre & Audience Perspective:
Please adjust your questioning angle and style according to the following genre and audience combinations:

**Target Genre**: "Journal Paper"
**Target Audience**: 
- Researchers and engineers in mechanics, materials science, physics, and related fields
- University faculty and graduate students in related majors
- Technical personnel engaged in metamaterial design, simulation, and experiments
- Scholars interested in novel material structures and properties

Please ensure:
1. Questions fully match the style, focus, and depth of the "Target Genre".
2. Questions consider the knowledge level, cognitive characteristics, and potential interests of the "Target Audience".
3. Raise questions from the perspective and needs of this audience group.
4. Maintain the relevance and practicality of the questions, ensuring consistency in question-answer style.
5. Questions should be clear and specific, avoiding overly broad or vague questions.

## Workflow
1. [Text Analysis] Process the content by paragraph, identify key entities and core concepts
2. [Question Generation] Select the best questioning points based on information density and combine with the specified genre and audience perspective
3. [Quality Check] Ensure:
   - The answer to the question can be found in the original text
   - Tags are highly relevant to the question content
   - No formatting errors
   - The question style matches the specified genre and audience

## Output Format
- JSON array format must be correct
- Field names must use English double quotes
- The output JSON array must strictly follow the structure below:
```json
["Question 1", "Question 2", "..."]
```

## Output Example
```json
["What are the core elements that should be included in the ethical framework for artificial intelligence?", "What new regulations on personal data protection are introduced in the Civil Code?"]
```

## Text to Process
{text}

## Restrictions
- Must output strictly in the specified JSON format, do not output any other irrelevant content
- Generate no fewer than {number} high-quality questions
- Do not generate questions related to the material itself, such as author, chapter, table of contents, etc.
- Questions must not contain phrases like [report, article, literature, table, formula, formula], and must be natural questions
"""


TXTQ2A_prompt = """
# Role: Fine-tuning Dataset Generation Expert
## Profile:
- Description: You are an expert in generating fine-tuning datasets in the field of mechanical metamaterials, skilled at generating accurate question answers from given content, and able to adjust answer style according to genre-audience combinations to ensure accuracy, relevance, and specificity.

## Skills:
1. Answers must be based on the given content
2. Answers must be accurate, no fabrication
3. Answers must be relevant to the question
4. Answers must be logical
5. Integrate the answer into a complete response in natural and fluent language based on the given reference content, without mentioning literature sources or citation marks
6. Able to adjust answer style and depth according to the specified genre and audience combination
7. While maintaining accuracy, enhance the specificity and applicability of the answer

**Target Genre**: "Journal Paper"
**Target Audience**: 
- Researchers and engineers in mechanics, materials science, physics, and related fields
- University faculty and graduate students in related majors
- Technical personnel engaged in metamaterial design, simulation, and experiments
- Scholars interested in novel material structures and properties

Please ensure:
1. The organization, style, level of detail, and language of the answer fully meet the requirements of the "Target Genre".
2. The answer should consider the comprehension and knowledge background of the target audience, striving for clarity and understandability.
3. Word choice and level of explanation should match the knowledge background of the target audience.
4. Maintain accuracy and professionalism while enhancing specificity.
5. If the "Target Genre" or audience suggests, the answer may appropriately include explanations, examples, or steps.
6. The answer should directly address the question, ensuring logical and coherent Q&A, and should not contain irrelevant information or citation marks as mentioned in GA to avoid data contamination.

## Workflow:
1. Take a deep breath and work on this problem step-by-step.
2. First, analyze the given file content and question type
3. Then, extract key information from the content
4. If a genre and audience combination is specified, analyze how to adjust the answer style
5. Next, generate an accurate answer relevant to the question, and adjust the expression according to the genre and audience requirements
6. Finally, ensure the accuracy, relevance, and style adaptation of the answer

## Reference Content:
{text}

## Question
{question}

## Constraints:
1. The answer must be based on the given content
2. The answer must be accurate, relevant to the question, and not fabricated
3. The answer must be sufficient, detailed, contain all necessary information, and be suitable for fine-tuning large model training
4. The answer must not contain any referential statements such as 'according to / based on / mentioned in the literature', only present the final result
5. If a genre and audience combination is specified, the answer style and depth must be adjusted while maintaining accuracy
6. The answer must directly address the question, ensuring accuracy and logic.
"""

OPTAnswer_prompt = """
# Role: Fine-tuning Dataset Answer Optimization Expert
## Profile:
- Description: You are an expert in optimizing answers for fine-tuning datasets, skilled at optimizing answers and reasoning processes (chain-of-thought) based on user improvement suggestions

## Skills:
1. Based on the given optimization advice + question, optimize the input answer, and appropriately enrich and supplement it
   
## Original Question
{question}

## Answer to Optimize
{answer}

## Answer Optimization Advice
{advice}, and appropriately enrich and supplement the answer to ensure accuracy, sufficiency, and clarity

## Constraints:
1. The result must be output in JSON format:
```json
{
"answer": "Optimized answer",
}
```
"""