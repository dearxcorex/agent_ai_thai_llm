"""
Local Model Integration for Thai Government Document Generation
Integrates context engineering with local LLM deployment
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

# Check for available local model libraries
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from context_engineering_system import ThaiDocumentContextEngineer, create_context_engineer
from utils.thai_formatting import ThaiFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalThaiDocumentGenerator:
    """Local model-based Thai document generator with context engineering"""

    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize local model generator

        Args:
            model_path: Path to local model or HuggingFace model name
            device: Device to run model on ("auto", "cpu", "cuda")
        """
        self.model_path = model_path or "sea-lion-7b-chat"  # Default to Thai model
        self.device = self._determine_device(device)
        self.context_engineer = create_context_engineer()
        self.thai_formatter = ThaiFormatter()

        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.langchain_llm = None

        # Load model
        self._initialize_model()

    def _determine_device(self, device: str) -> str:
        """Determine the best device for model execution"""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif TORCH_AVAILABLE and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device

    def _initialize_model(self):
        """Initialize the local model"""
        try:
            if TRANSFORMERS_AVAILABLE:
                logger.info(f"Loading model: {self.model_path} on device: {self.device}")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=False
                )

                # Set padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model with optimized settings
                model_kwargs = {
                    "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                    "device_map": "auto" if self.device != "cpu" else None,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True
                }

                if self.device == "cpu":
                    model_kwargs.pop("device_map", None)
                    model_kwargs["device"] = "cpu"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )

                # Create generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Create LangChain wrapper if available
                if LANGCHAIN_AVAILABLE:
                    self.langchain_llm = HuggingFacePipeline(pipeline=self.pipeline)

                logger.info("Model loaded successfully")

            else:
                logger.error("Transformers library not available. Install with: pip install transformers torch")
                raise ImportError("Required libraries not available")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def generate_document(self, user_query: str, context_documents: List[str] = None,
                         max_tokens_per_section: int = 512) -> Dict:
        """
        Generate complete Thai government document

        Args:
            user_query: User's document request
            context_documents: Additional context documents
            max_tokens_per_section: Maximum tokens per document section

        Returns:
            Generated document with metadata
        """
        try:
            logger.info(f"Generating document for query: {user_query[:50]}...")

            # Step 1: Classify document type
            document_type, confidence = self.context_engineer.classify_request(user_query)
            logger.info(f"Document classified as: {document_type} (confidence: {confidence:.2f})")

            # Step 2: Extract structured information
            extracted_info = self.context_engineer.extract_document_info(user_query)
            logger.info(f"Extracted info: {len(extracted_info)} categories")

            # Step 3: Generate document sections
            document_sections = self._generate_document_sections(
                document_type, extracted_info, user_query, context_documents, max_tokens_per_section
            )

            # Step 4: Apply Thai formatting
            formatted_document = self._apply_thai_formatting(document_sections, document_type)

            # Step 5: Validate and quality check
            validation_result = self._validate_document(formatted_document, document_type)

            # Calculate quality score
            quality_score = self._calculate_quality_score(validation_result, confidence)

            logger.info(f"Document generated successfully (score: {quality_score:.2f})")

            return {
                "document": formatted_document,
                "document_type": document_type,
                "confidence": confidence,
                "extracted_info": extracted_info,
                "sections": document_sections,
                "validation": validation_result,
                "quality_score": quality_score,
                "generation_metadata": {
                    "model_path": self.model_path,
                    "device": self.device,
                    "max_tokens_per_section": max_tokens_per_section,
                    "generated_at": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            return {
                "document": f"เกิดข้อผิดพลาดในการสร้างเอกสาร: {str(e)}",
                "error": str(e),
                "quality_score": 0.0
            }

    def _generate_document_sections(self, document_type: str, extracted_info: Dict,
                                  user_query: str, context_documents: List[str],
                                  max_tokens: int) -> Dict[str, str]:
        """Generate individual document sections using context engineering"""

        # Get document structure
        template = self.context_engineer.document_templates.get(document_type)
        if not template:
            template = self.context_engineer.document_templates["บันทึกข้อความ"]

        sections = {}

        # Generate each section
        for section_config in template["structure"]:
            section_name = section_config["section"]

            # Skip optional sections if not enough information
            if not section_config["required"] and not self._should_include_section(section_name, extracted_info):
                continue

            # Create context for this section
            context = self.context_engineer.create_context_template(
                document_type, section_name, extracted_info
            )

            # Generate context prompt
            context_prompt = self.context_engineer.generate_context_prompt(context)

            # Optimize for local model
            optimized_prompt = self.context_engineer.optimize_context_for_local_model(
                context_prompt, max_tokens
            )

            # Generate content for this section
            section_content = self._generate_section_content(
                optimized_prompt, user_query, context_documents, max_tokens
            )

            sections[section_name] = section_content
            logger.info(f"Generated section: {section_name} ({len(section_content)} chars)")

        return sections

    def _should_include_section(self, section_name: str, extracted_info: Dict) -> bool:
        """Determine if optional section should be included"""

        section_rules = {
            "เรื่องเพื่อทราบ": lambda info: any(info.values()),  # Include if any info exists
            "ผู้เข้าร่วม": lambda info: info.get("people"),  # Include if people mentioned
            "วาระการประชุม": lambda info: "ประชุม" in " ".join(info.get("action_words", [])),
            "ข้อเสนอแนะ": lambda info: "ตรวจสอบ" in " ".join(info.get("action_words", []))
        }

        if section_name in section_rules:
            return section_rules[section_name](extracted_info)

        return True

    def _generate_section_content(self, context_prompt: str, user_query: str,
                                context_documents: List[str], max_tokens: int) -> str:
        """Generate content for a specific section using local model"""

        # Create complete prompt
        full_prompt = f"""{context_prompt}

คำขอจากผู้ใช้:
{user_query}

เอกสารอ้างอิง (ถ้ามี):
{" ".join(context_documents[:2]) if context_documents else "ไม่มี"}

กรุณาสร้างเนื้อหาสำหรับส่วนที่ระบุตามโครงสร้างและข้อกำหนดข้างต้น:
"""

        try:
            if self.langchain_llm:
                # Use LangChain wrapper if available
                result = self.langchain_llm(full_prompt)
            else:
                # Use transformers pipeline directly
                result = self.pipeline(full_prompt)[0]["generated_text"]

            # Extract only the generated part (remove prompt)
            if full_prompt in result:
                content = result.replace(full_prompt, "").strip()
            else:
                content = result.strip()

            # Clean up the content
            content = self._clean_generated_content(content)

            return content

        except Exception as e:
            logger.error(f"Section generation failed: {e}")
            return f"[เกิดข้อผิดพลาดในการสร้างเนื้อหา: {str(e)}]"

    def _clean_generated_content(self, content: str) -> str:
        """Clean and format generated content"""

        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        # Ensure proper Thai numbering
        content = self.thai_formatter.apply_thai_formatting(content)

        # Fix common formatting issues
        content = self._fix_common_formatting_issues(content)

        return content.strip()

    def _fix_common_formatting_issues(self, text: str) -> str:
        """Fix common formatting issues in generated text"""

        # Fix section numbering
        text = re.sub(r'(\d+)\.', lambda m: self.thai_formatter.convert_to_thai_numbers(m.group(1)) + '.', text)

        # Fix date formats
        text = re.sub(r'(\d{4})', lambda m: str(int(m.group(1)) + 543) if int(m.group(1)) < 2500 else m.group(1), text)

        # Add proper spacing after punctuation
        text = re.sub(r'([.:,])([^\s])', r'\1 \2', text)

        return text

    def _apply_thai_formatting(self, sections: Dict[str, str], document_type: str) -> str:
        """Apply Thai government formatting to complete document"""

        # Get document header
        header = self._generate_document_header(document_type)

        # Combine sections
        document_parts = [header]

        # Add sections in correct order
        template = self.context_engineer.document_templates.get(document_type)
        if template:
            for section_config in template["structure"]:
                section_name = section_config["section"]
                if section_name in sections:
                    document_parts.append(sections[section_name])

        # Add footer
        footer = self._generate_document_footer(document_type)
        document_parts.append(footer)

        # Apply final formatting
        full_document = "\n\n".join(document_parts)
        formatted_document = self.thai_formatter.apply_thai_formatting(full_document)

        return formatted_document

    def _generate_document_header(self, document_type: str) -> str:
        """Generate document header"""

        # Current date in Thai format
        current_date = self.thai_formatter.format_thai_date(
            datetime.now().strftime("%d %B %Y")
        )

        # Document number
        doc_number = f"สทช {datetime.now().strftime('%m%d')}/{datetime.now().strftime('%Y')}"

        # Organization
        org = "สำนักงานคณะกรรมการกิจการกระจายเสียง กิจการโทรทัศน์ และกิจการโทรคมนาคมแห่งชาติ"

        header_parts = [
            f"{org}",
            f"ที่ {doc_number}",
            f"วันที่ {current_date}"
        ]

        return "\n".join(header_parts)

    def _generate_document_footer(self, document_type: str) -> str:
        """Generate document footer"""

        footer_parts = [
            "จึงเรียนมาเพื่อโปรดพิจารณา" if "บันทึก" in document_type else "จึงเรียนมาเพื่อทราบ",
            "",
            "(ลงชื่อ)",
            ".................................................",
            "ผู้อำนวยการสำนักงาน กสทช."
        ]

        return "\n".join(footer_parts)

    def _validate_document(self, document: str, document_type: str) -> Dict:
        """Validate generated document"""

        validation_result = {
            "is_valid": True,
            "issues": [],
            "score": 1.0
        }

        # Check required elements
        required_elements = self.context_engineer.document_templates.get(document_type, {}).get("required_elements", [])

        for element in required_elements:
            if element.lower() not in document.lower():
                validation_result["issues"].append({
                    "type": "missing_element",
                    "element": element,
                    "severity": "error"
                })
                validation_result["score"] -= 0.2

        # Check formatting
        if not re.search(r'[๑-๙]', document):  # Thai numerals check
            validation_result["issues"].append({
                "type": "formatting",
                "issue": "ไม่พบเลขไทยในเอกสาร",
                "severity": "warning"
            })
            validation_result["score"] -= 0.1

        # Check for proper sections
        expected_sections = ["เรื่องเดิม", "เรื่องเพื่อพิจารณา"]
        if document_type == "บันทึกข้อความ":
            for section in expected_sections:
                if section not in document:
                    validation_result["issues"].append({
                        "type": "missing_section",
                        "section": section,
                        "severity": "error"
                    })
                    validation_result["score"] -= 0.15

        validation_result["is_valid"] = validation_result["score"] >= 0.7
        validation_result["score"] = max(0.0, validation_result["score"])

        return validation_result

    def _calculate_quality_score(self, validation_result: Dict, classification_confidence: float) -> float:
        """Calculate overall quality score"""

        validation_score = validation_result["score"]
        confidence_score = classification_confidence

        # Weighted average
        overall_score = (validation_score * 0.7) + (confidence_score * 0.3)

        return round(overall_score, 2)

    def test_model_performance(self, test_queries: List[str]) -> Dict:
        """Test model performance with sample queries"""

        results = []

        for i, query in enumerate(test_queries, 1):
            logger.info(f"Testing query {i}: {query[:50]}...")

            start_time = datetime.now()

            try:
                result = self.generate_document(query)
                generation_time = (datetime.now() - start_time).total_seconds()

                test_result = {
                    "query": query,
                    "success": not result.get("error"),
                    "generation_time": generation_time,
                    "quality_score": result.get("quality_score", 0),
                    "document_type": result.get("document_type"),
                    "document_length": len(result.get("document", "")),
                    "validation_issues": len(result.get("validation", {}).get("issues", []))
                }

                results.append(test_result)

            except Exception as e:
                logger.error(f"Test query {i} failed: {e}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "generation_time": (datetime.now() - start_time).total_seconds()
                })

        # Calculate performance metrics
        successful_tests = [r for r in results if r.get("success", False)]

        performance_metrics = {
            "total_tests": len(test_queries),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(test_queries) * 100,
            "average_generation_time": sum(r.get("generation_time", 0) for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            "average_quality_score": sum(r.get("quality_score", 0) for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            "average_document_length": sum(r.get("document_length", 0) for r in successful_tests) / len(successful_tests) if successful_tests else 0
        }

        return {
            "performance_metrics": performance_metrics,
            "detailed_results": results,
            "model_info": {
                "model_path": self.model_path,
                "device": self.device,
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
                "langchain_available": LANGCHAIN_AVAILABLE
            }
        }

# Factory function
def create_local_generator(model_path: str = None, device: str = "auto") -> LocalThaiDocumentGenerator:
    """Create local document generator instance"""
    return LocalThaiDocumentGenerator(model_path, device)