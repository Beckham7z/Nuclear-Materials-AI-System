#!/usr/bin/env python3
"""
Nuclear Material Domain Expert Model - Data Cleaning Pipeline
============================================================
This script cleans and transforms LightRAG JSON data for fine-tuning:
1. Removes PDF artifacts (page markers, image placeholders, etc.)
2. Anonymizes personal names and place names
3. Standardizes QA pair formats
4. Converts to ShareGPT format for SFT training

Author: Nuclear Material AI/ML Pipeline
Date: 2026/03/08
"""

import json
import re
import os
import glob
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CleaningConfig:
    """Configuration for data cleaning pipeline"""
    # Input/Output
    input_pattern: str = "nuclear_qa_v3_part_*.json"
    output_dir: str = "cleaned_data"
    
    # Content cleaning
    remove_pdf_markers: bool = True
    remove_references: bool = True
    remove_image_placeholders: bool = True
    normalize_whitespace: bool = True
    
    # Anonymization
    anonymize_names: bool = True
    anonymize_locations: bool = True
    
    # QA processing
    standardize_format: bool = True
    min_answer_length: int = 20
    max_answer_length: int = 2000
    
    # Output format
    output_format: str = "sharegpt"  # "sharegpt" or "alpaca"


# Common patterns to remove (PDF artifacts)
PDF_PATTERNS = {
    # Page markers
    r"##\s*第\s*\d+\s*页": "",
    r"##\s*Page\s*\d+": "",
    r"Page\s+\d+\s+of\s+\d+": "",
    
    # Image placeholders
    r"!\[图片[^\]]*\]\([^)]+\)": "",
    r"!\[Figure[^\]]*\]\([^)]+\)": "",
    r"\[图片\s*\d+-\d+\]": "",
    r"\[Figure\s*\d+\]": "",
    
    # Separators
    r"^---+$": "",
    r"^\*\*\*+$": "",
    
    # Reference markers
    r"\[\d+\]": "",
    r"\[REF\d+\]": "",
    r"\[Citation\s+\d+\]": "",
    
    # ArXiv/Preprint markers
    r"arXiv:\d+\.\d+v\d+.*?\d+\s+\w+\s+\d{4}": "",
    
    # DOI links
    r"https?://doi\.org/[^\s]+": "",
}


# Names and locations to anonymize (extensible)
# In production, use NER models like spaCy or transformers
SENSITIVE_NAMES = [
    # Common first names that might appear
    r"\b(John|Jane|Alice|Bob|Charlie|David|Eve|Mary|Peter|Tom|Mike|Smith|Johnson|Williams|Brown|Jones|Garcia|Miller|Davis|Rodriguez|Martinez)\b",
]

SENSITIVE_LOCATIONS = [
    # Institutions that might need anonymization
    r"(University of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    r"(Institute of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    r"(National\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    r"(Lawrence\s+Berkeley|Argonne|Los\s+Alamos|Oak\s+Ridge|LANL|ORNL|ANL)",
    r"(Beijing|Shanghai|Shenzhen|Hangzhou|Chengdu|Wuhan|Xian)",
    r"(Germany|France|Japan|Korea|Russia|China|USA)",
]


# ============================================================================
# Core Cleaning Functions
# ============================================================================

class ContentCleaner:
    """Cleans content field of PDF artifacts and noise"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
    
    def clean(self, text: str) -> str:
        """Main cleaning function"""
        if not text:
            return ""
        
        result = text
        
        if self.config.remove_pdf_markers:
            result = self._remove_pdf_markers(result)
        
        if self.config.remove_image_placeholders:
            result = self._remove_image_placeholders(result)
        
        if self.config.remove_references:
            result = self._remove_references(result)
        
        if self.config.normalize_whitespace:
            result = self._normalize_whitespace(result)
        
        return result.strip()
    
    def _remove_pdf_markers(self, text: str) -> str:
        """Remove PDF page markers and similar artifacts"""
        # Remove page headers
        text = re.sub(r"##\s*第\s*\d+\s*页.*", "", text)
        text = re.sub(r"##\s*Page\s*\d+.*", "", text)
        
        # Remove preprint/arxiv markers
        text = re.sub(r"PRE-PRINT.*?\d{4}", "", text)
        text = re.sub(r"arXiv:\d+\.\d+v\d+.*?\d+\s+\w+\s+\d{4}", "", text)
        
        # Remove Elsevier/header markers
        text = re.sub(r"Preprint submitted to.*?\.?\s*$", "", text, flags=re.MULTILINE)
        
        return text
    
    def _remove_image_placeholders(self, text: str) -> str:
        """Remove image/figure placeholders"""
        # Markdown image syntax
        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[Figure: \1]", text)
        
        # Raw image markers
        text = re.sub(r"\[图片\s*\d+[^\]]*\]", "", text)
        text = re.sub(r"\[Figure\s*\d+[^\]]*\]", "", text)
        
        return text
    
    def _remove_references(self, text: str) -> str:
        """Remove reference numbers and citations"""
        # Remove [1], [2], etc.
        text = re.sub(r"\[\d+(,\s*\d+)*\]", "", text)
        
        # Remove (Author, Year) style citations
        text = re.sub(r"\([A-Z][a-z]+.*?\d{4}\)", "", text)
        
        # Remove URLs
        text = re.sub(r"https?://[^\s]+", "", text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and remove extra newlines"""
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        
        # Remove trailing whitespace on each line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text


class Anonymizer:
    """Anonymizes sensitive information like names and locations"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.replacement_map: Dict[str, str] = {}
    
    def anonymize(self, text: str) -> str:
        """Main anonymization function"""
        if not text:
            return ""
        
        result = text
        
        if self.config.anonymize_names:
            result = self._anonymize_names(result)
        
        if self.config.anonymize_locations:
            result = self._anonymize_locations(result)
        
        return result
    
    def _anonymize_names(self, text: str) -> str:
        """Replace personal names with generic placeholders"""
        # Replace author names in acknowledgments/funders
        text = re.sub(
            r"(Author|Authors|Professor|Dr\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
            "[Researcher]",
            text
        )
        
        # Replace email addresses
        text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", 
                     "[Email]", text)
        
        # Replace funding agencies and project numbers
        text = re.sub(r"(U\.S\.\s+Department of Energy|DOE|NSF|NASA|DARPA).*?(?:grant|contract|project).*?\d+",
                     "[Funding Agency]", text, flags=re.IGNORECASE)
        
        return text
    
    def _anonymize_locations(self, text: str) -> str:
        """Replace sensitive locations with generic placeholders"""
        # Replace specific institution names
        institutions = [
            (r"Lawrence\s+Berkeley\s+National\s+Laboratory", "[National Laboratory]"),
            (r"Los\s+Alamos\s+National\s+Laboratory", "[National Laboratory]"),
            (r"Oak\s+Ridge\s+National\s+Laboratory", "[National Laboratory]"),
            (r"Argonne\s+National\s+Laboratory", "[National Laboratory]"),
            (r"China\s+Academy\s+of\s+Engineering\s+Physics", "[Research Institute]"),
        ]
        
        for pattern, replacement in institutions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Replace specific city names (except well-known nuclear research centers)
        # Keep general country names, remove specific cities
        
        return text


class QAPairProcessor:
    """Processes and standardizes QA pairs"""
    
    # Core nuclear material science keywords (must have at least one)
    NUCLEAR_MATERIAL_KEYWORDS = [
        # Materials
        "hastelloy", "zirconium", "uranium", "plutonium", "thorium", "neutron",
        "alloy", "steel", " cladding", "fuel", "reactor", "radiation",
        "corrosion", "oxidation", "embrittlement", "irradiation", "dose",
        "material", "metall", "ceramic", "cermet", "composite", "coating",
        # Properties
        "mechanical", "thermal", " creep", "fatigue", "hardness", "tensile",
        "fracture", "ductility", "strength", " toughness", "elastic",
        "microstructure", "phase", "grain", "defect", "vacancy", "dislocation",
        # Nuclear specific
        "fission", "fusion", "breeder", "spent fuel", "reprocessing",
        "tritium", "helium", "transmutation", "activation", "half-life",
        "moderator", "coolant", "reflector", "shield", "absorber",
        "molten salt", "fluoride", "chloride", "liquid metal",
        # Analysis methods
        "sem", "tem", "xrd", "xps", "aes", "stxm", "xtas", "stem",
        "eds", "wds", "epma", "aft", "sims", "atom probe",
    ]
    
    # Topics to filter out (not relevant to nuclear material design)
    FILTER_OUT_TOPICS = [
        # Medical applications (not material design)
        "medical treatment", "diagnostic", "therapy", "radiotherapy",
        "cancer", "tumor", "patient", "clinical", "drug delivery",
        # Astrophysics (not material design) - too broad, keep nuclear相关内容
        # "star", "supernova", "neutron star", "big bang", "cosmology",
        # Non-nuclear optics/electronics - too broad
        # "optical fiber", "photonic", "laser cavity", "bragg grating",
        # Network/detection (not material design) - too broad
        # "network", "detection network", "data fusion", "mobile source",
        # General topics not specific to nuclear materials
        "climate change", "economic analysis", "policy", "regulation",
        "cost estimate", "safety standard", "licensing",
    ]
    
    # Additional check: require at least one core keyword from question itself
    CORE_QUESTION_KEYWORDS = [
        "material", "alloy", "corrosion", "fuel", "cladding", "reactor",
        "uranium", "plutonium", "zirconium", "hastelloy", "metal",
        "ceramic", "radiation", "irradiation", "neutron", "temperature",
        "mechanical", "thermal", "phase", "microstructure", "coating",
        "embrittlement", "oxidation", "creep", "fatigue", "fracture",
    ]
    
    def __init__(self, config: CleaningConfig):
        self.config = config
    
    def process(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Process list of QA pairs"""
        if not qa_pairs:
            return []
        
        processed = []
        for qa in qa_pairs:
            cleaned = self._process_single(qa)
            if cleaned:
                processed.append(cleaned)
        
        return processed
    
    def _process_single(self, qa: Dict) -> Optional[Dict]:
        """Process single QA pair"""
        # Extract question
        question = self._extract_question(qa)
        if not question:
            return None
        
        # Extract answer
        answer = self._extract_answer(qa)
        if not answer:
            return None
        
        # Filter out Chinese content - keep only English QA pairs
        if self._contains_chinese(question) or self._contains_chinese(answer):
            return None
        
        # Filter out non-nuclear-material topics
        if not self._is_relevant_to_nuclear_materials(question, answer):
            return None
        
        # Validate lengths
        if len(answer) < self.config.min_answer_length:
            return None
        
        if len(answer) > self.config.max_answer_length:
            answer = answer[:self.config.max_answer_length] + "..."
        
        return {
            "question": question,
            "answer": answer
        }
    
    def _is_relevant_to_nuclear_materials(self, question: str, answer: str) -> bool:
        """Check if QA pair is relevant to nuclear material design"""
        combined = (question + " " + answer).lower()
        
        # First check if it matches any filter-out topics
        for topic in self.FILTER_OUT_TOPICS:
            if topic.lower() in combined:
                return False
        
        # Then check if it contains at least one nuclear material keyword
        keyword_count = 0
        for keyword in self.NUCLEAR_MATERIAL_KEYWORDS:
            if keyword.lower() in combined:
                keyword_count += 1
                if keyword_count >= 1:  # At least one keyword
                    return True
        
        return False
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        if not text:
            return False
        # Check for Chinese characters (CJK Unified Ideographs)
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
                return True
            if '\u3400' <= char <= '\u4dbf':  # CJK Unified Ideographs Extension A
                return True
            if '\U00020000' <= char <= '\U0002a6df':  # CJK Unified Ideographs Extension B
                return True
        return False
    
    def _extract_question(self, qa: Dict) -> str:
        """Extract question from various formats"""
        # Try different keys
        for key in ["question", "Question", "q", "Q"]:
            if key in qa:
                value = qa[key]
                if isinstance(value, str):
                    # Handle dict format: {'number': 1, 'question': '...'}
                    try:
                        parsed = json.loads(value) if value.startswith('{') else value
                        if isinstance(parsed, dict):
                            return parsed.get("question", str(value))
                    except:
                        pass
                    return value
        
        return str(qa.get("question", ""))
    
    def _extract_answer(self, qa: Dict) -> str:
        """Extract answer from various formats"""
        # Prefer optimized_answer if available
        for key in ["optimized_answer", "answer", "Answer"]:
            if key in qa:
                value = qa[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
        
        return ""


class ShareGPTConverter:
    """Converts QA pairs to ShareGPT format"""
    
    SYSTEM_PROMPT = """You are a helpful assistant specializing in nuclear material science. 
Provide accurate, professional answers based on scientific knowledge."""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
    
    def convert(self, qa_pairs: List[Dict], doc_content: str = "") -> List[Dict]:
        """Convert QA pairs to ShareGPT format"""
        results = []
        
        for qa in qa_pairs:
            # Extract keywords/topic from content for context
            topic = self._extract_topic(doc_content, qa.get("question", ""))
            
            conversation = {
                "conversations": [
                    {
                        "from": "system",
                        "value": self.SYSTEM_PROMPT
                    },
                    {
                        "from": "user", 
                        "value": self._format_question(qa.get("question", ""), topic)
                    },
                    {
                        "from": "assistant",
                        "value": qa.get("answer", "")
                    }
                ]
            }
            
            results.append(conversation)
        
        return results
    
    def _extract_topic(self, content: str, question: str) -> str:
        """Extract topic from content for better context"""
        if not content:
            return ""
        
        # Get first 500 chars as context
        return content[:500].replace("\n", " ").strip()
    
    def _format_question(self, question: str, topic: str) -> str:
        """Format question with optional context - clean up nested dict format"""
        # Clean up question format: {'number': 1, 'question': '...'} -> just the question
        question = self._clean_question_format(question)
        
        if topic:
            return f"[Context: {topic}]\n\nQuestion: {question}"
        return question
    
    def _clean_question_format(self, question: str) -> str:
        """Clean up question format by removing dict wrappers"""
        # Handle format like {'number': 1, 'question': '...'}
        if question.startswith("{"):
            try:
                parsed = json.loads(question)
                if isinstance(parsed, dict):
                    if "question" in parsed:
                        return parsed["question"]
                    # Also check for other keys
                    for key in ["q", "Question", "Q"]:
                        if key in parsed:
                            return str(parsed[key])
            except json.JSONDecodeError:
                pass
        
        # Also handle quoted dict strings
        question = re.sub(r"^\{'[^']+':\s*[^,]+,\s*'question':\s*'", "", question)
        question = re.sub(r"^\{'question':\s*'", "", question)
        question = re.sub(r"'\s*\}$", "", question)
        
        return question


class AlpacaConverter:
    """Converts QA pairs to Alpaca format"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
    
    def convert(self, qa_pairs: List[Dict], doc_content: str = "") -> List[Dict]:
        """Convert QA pairs to Alpaca format"""
        results = []
        
        for qa in qa_pairs:
            item = {
                "instruction": qa.get("question", ""),
                "input": "",
                "output": qa.get("answer", "")
            }
            results.append(item)
        
        return results


# ============================================================================
# Main Pipeline
# ============================================================================

class DataCleaningPipeline:
    """Main data cleaning pipeline"""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        
        # Initialize components
        self.content_cleaner = ContentCleaner(self.config)
        self.anonymizer = Anonymizer(self.config)
        self.qa_processor = QAPairProcessor(self.config)
        
        # Select converter based on output format
        if self.config.output_format == "sharegpt":
            self.converter = ShareGPTConverter(self.config)
        else:
            self.converter = AlpacaConverter(self.config)
    
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a single JSON file"""
        print(f"Processing: {input_path}")
        
        # Load data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        # Process each document
        processed_docs = []
        stats = {
            "total": len(data),
            "success": 0,
            "skipped": 0,
            "errors": 0
        }
        
        for idx, doc in enumerate(data):
            try:
                result = self._process_document(doc)
                if result:
                    processed_docs.append(result)
                    stats["success"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                print(f"  Error processing doc {idx}: {e}")
                stats["errors"] += 1
        
        # Save output
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_docs, f, ensure_ascii=False, indent=2)
            print(f"Saved to: {output_path}")
        
        print(f"Stats: {stats}")
        return {"data": processed_docs, "stats": stats}
    
    def _process_document(self, doc: Dict) -> Optional[Dict]:
        """Process single document - supports both formats"""
        
        # Format 1: Original format with content + qa_pairs array
        if "qa_pairs" in doc and "content" in doc:
            content = doc.get("content", "")
            if not content:
                return None
            
            # Clean content
            content = self.content_cleaner.clean(content)
            content = self.anonymizer.anonymize(content)
            
            # Process QA pairs
            qa_pairs = doc.get("qa_pairs", [])
            if not qa_pairs:
                return None
            
            processed_qa = self.qa_processor.process(qa_pairs)
            if not processed_qa:
                return None
            
            # Convert to target format
            converted = self.converter.convert(processed_qa, content)
            
            return {
                "id": doc.get("_id", doc.get("id", "")),
                "content": content,
                "qa_pairs": converted,
                "metadata": {
                    "source": doc.get("file_path", "unknown"),
                    "tokens": doc.get("tokens", 0)
                }
            }
        
        # Format 2: nuclear_qa_300.json format with content_preview, question, answer, optimized_answer
        elif "content_preview" in doc or "question" in doc:
            # Get content from content_preview or use empty string
            content = doc.get("content_preview", "")
            
            # Clean content
            content = self.content_cleaner.clean(content)
            content = self.anonymizer.anonymize(content)
            
            # Create QA pair from document fields
            qa = {
                "question": doc.get("question", ""),
                "answer": doc.get("optimized_answer", "") or doc.get("answer", "")
            }
            
            # Process single QA pair
            processed_qa = self.qa_processor.process([qa])
            if not processed_qa:
                return None
            
            # Convert to target format
            converted = self.converter.convert(processed_qa, content)
            
            return {
                "id": doc.get("text_id", doc.get("full_doc_id", "")),
                "content": content,
                "qa_pairs": converted,
                "metadata": {
                    "source": doc.get("file_path", "unknown"),
                    "tokens": doc.get("tokens", 0)
                }
            }
        
        return None
    
    def process_directory(self, input_pattern: str, output_dir: str) -> None:
        """Process all files matching pattern"""
        os.makedirs(output_dir, exist_ok=True)
        
        files = glob.glob(input_pattern)
        print(f"Found {len(files)} files to process")
        
        all_results = []
        total_stats = {"total": 0, "success": 0, "skipped": 0, "errors": 0}
        
        for filepath in files:
            filename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, f"cleaned_{filename}")
            
            result = self.process_file(filepath, output_path)
            all_results.extend(result.get("data", []))
            
            # Update stats
            for key in total_stats:
                total_stats[key] += result.get("stats", {}).get(key, 0)
        
        # Save combined output
        combined_path = os.path.join(output_dir, "combined_training_data.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Total Stats: {total_stats}")
        print(f"Combined data saved to: {combined_path}")
        
        # Also save as JSONL for easier processing
        jsonl_path = os.path.join(output_dir, "training_data.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"JSONL data saved to: {jsonl_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_hash(text: str) -> str:
    """Create short hash for deduplication"""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def validate_output(data: List[Dict]) -> Tuple[int, int]:
    """Validate output data quality"""
    valid = 0
    invalid = 0
    
    for item in data:
        qa_pairs = item.get("qa_pairs", [])
        if qa_pairs and all(
            "conversations" in qa and len(qa["conversations"]) == 3
            for qa in qa_pairs
        ):
            valid += 1
        else:
            invalid += 1
    
    return valid, invalid


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nuclear Material Data Cleaning Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        default="nuclear_qa_v3_part_*.json",
        help="Input file pattern"
    )
    parser.add_argument(
        "--output", "-o",
        default="cleaned_data",
        help="Output directory"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["sharegpt", "alpaca"],
        default="sharegpt",
        help="Output format"
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Skip name/location anonymization"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = CleaningConfig(
        input_pattern=args.input,
        output_dir=args.output,
        output_format=args.format,
        anonymize_names=not args.no_anonymize,
        anonymize_locations=not args.no_anonymize
    )
    
    # Run pipeline
    pipeline = DataCleaningPipeline(config)
    pipeline.process_directory(args.input, args.output)
    
    print("\n✅ Data cleaning complete!")


if __name__ == "__main__":
    main()
