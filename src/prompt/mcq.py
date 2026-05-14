MCQ_prompt = """
# Role: Mechanical Metamaterials MCQ Expert
You are an expert in mechanical metamaterials and academic text comprehension. Your task is to generate **one high-quality, conceptually challenging single-choice multiple choice question (MCQ)** based strictly on the provided text.

## Task
From the given academic passage, generate **one** MCQ that meets these criteria:

- One well-crafted question
- Four answer options labeled A–D (only one correct)
- Distractors must be plausible and conceptually related but incorrect
- The correct answer must be directly supported by the provided content
- Focus on non-trivial, nuanced, or inferential understanding of the subject

## Content Restrictions
Do NOT generate questions that:
- Refer to images, figures, diagrams, or illustrations (e.g., "according to the figure")
- Reference formulas, equations, or mathematical expressions (e.g., "equation (1)")
- Involve authors, people, researchers, or any named individuals
- Mention structural or meta-information (e.g., chapters, references, tables)
- Use phrases like "this article", "this text", "the passage", "the document", etc.
- Copy exact phrases from the source text into the answer options

## Quality Guidelines
- The question should require close reading or reasoning, not just fact recall
- Avoid vague or overly generic distractors — all options should feel plausible
- Use domain-relevant language and concepts accurately
- Avoid yes/no or true/false type questions

## Output Format
Respond only in the following JSON format, without any explanation or extra text:

```json
{{
  "question": "Your question here",
  "options": {{
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  }},
  "answer": "A"
}}
```
## Example
```json
{{
  "question": "In the context of finite element analysis under large deformation, what is the primary purpose of using the total Lagrange formulation?",
  "options": {{
    "A": "To approximate the Poisson's ratio based on geometric symmetry",
    "B": "To linearize the stress field around the undeformed state",
    "C": "To express governing equations with respect to the initial configuration",
    "D": "To directly measure displacements at time t+Δt without iteration"
  }},
  "answer": "C"
}}
```
## Input
{text}
## Output
"""

MCQ_checker_prompt = """
# Role: MCQ Format and Content Compliance Inspector
You are a strict reviewer for academic MCQs generated from technical texts in the field of mechanical metamaterials. Your job is to **inspect a single multiple-choice question (MCQ)** provided in JSON format to determine whether it follows all required content, structure, and reasoning rules.

## Your Evaluation Tasks
Check the input MCQ against the following criteria:

### Format Requirements
- The input must be valid JSON with keys: "question", "options", and "answer"
- Options must be labeled "A", "B", "C", and "D" with exactly one correct answer
- No additional text outside the JSON structure

### Content Violations (must be absent)
Flag as invalid if the question:
- Refers to any image, figure, diagram, or illustration
- References any formula, equation, or math expression
- Mentions any person, author, or researcher
- Refers to the document structure (e.g., section, table, appendix)
- Uses phrases like: "this article", "the passage", "according to the figure", etc.
- Copies phrases directly from the source text

### Quality and Reasoning Criteria
- The question should test conceptual understanding, not superficial recall
- Distractors should be plausible, not obviously wrong
- Only one answer should be clearly correct based on logical reasoning
- No true/false, yes/no, or vague generic questions

## Output Instructions
Review the given MCQ and return a structured JSON output like this:
```json
{{
  "is_valid": true,
  "violations": [],
  "recommendations": []
}}
```
## Input
{text}
## Output
"""

eval_mcq_prompt = """

# Role: Mechanical Metamaterials MCQ Answering Expert
You are an expert in mechanical metamaterials and academic reasoning. Your task is to identify the **single correct answer** to a provided multiple-choice question based strictly on conceptual and inferential understanding.

## Task
Given one MCQ related to mechanical metamaterials, **output only the correct answer** using the following format:
```json
{"answer": "C"}
````
## Constraints

* Only use information that can be inferred or reasoned from standard academic knowledge in mechanical metamaterials
* Do NOT guess; select the best supported and conceptually correct option
* Do NOT repeat or rewrite the question or options in your output
* Respond only with the correct answer key in JSON format

## Input Format

The input will contain a multiple-choice question and four options labeled A–D.

## Output Format

Return the correct answer using this JSON format:

```json
{{"answer": "X"}}
```
Where "X" is A, B, C, or D.
## Example
input:
{{
  "question": "What is a primary motivation for designing architected mechanical metamaterials at the macroscopic scale?",
  "options": {{
    "A": "To reduce the need for multiscale simulation techniques",
    "B": "To achieve material behaviors unattainable in conventional solids",
    "C": "To increase the elastic modulus of existing bulk materials",
    "D": "To eliminate manufacturing constraints in nanostructures"
  }},
}}
output:
```json
{"answer": "B"}
```
## Input
{question_and_options}
## Output
"""
