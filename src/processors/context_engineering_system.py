"""
Context Engineering System for Thai Government Document Generation
Optimized for local model deployment with enhanced accuracy
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class DocumentContext:
    """Context information for document generation"""
    document_type: str
    section: str
    formatting_rules: List[str]
    content_requirements: Dict
    language_patterns: List[str]
    examples: Optional[List[str]] = None

@dataclass
class GenerationRequest:
    """Request structure for document generation"""
    user_query: str
    document_type: str
    priority_sections: List[str]
    technical_terms: List[str]
    recipients: List[str]
    references: List[str]

class ThaiDocumentContextEngineer:
    """Context engineering system for Thai government documents"""

    def __init__(self):
        self.document_templates = self._load_document_templates()
        self.formatting_rules = self._load_formatting_rules()
        self.language_patterns = self._load_language_patterns()

    def _load_document_templates(self) -> Dict:
        """Load document type templates"""
        return {
            "บันทึกข้อความ": {
                "structure": [
                    {"section": "หัวข้อเอกสาร", "required": True, "order": 1},
                    {"section": "เรื่องเดิม", "required": True, "order": 2},
                    {"section": "เรื่องเพื่อพิจารณา", "required": True, "order": 3},
                    {"section": "เรื่องเพื่อทราบ", "required": False, "order": 4},
                    {"section": "ลงนาม", "required": True, "order": 5}
                ],
                "formatting": {
                    "numbering": "thai_numerals",
                    "date_format": "buddhist_calendar",
                    "language_style": "formal_official"
                },
                "required_elements": ["เลขที่", "วันที่", "เรื่อง", "เรียน"],
                "keywords": ["บันทึก", "ข้อความ", "ทะเบียน", "รายงาน"]
            },
            "หนังสือเชิญ": {
                "structure": [
                    {"section": "หัวข้อเอกสาร", "required": True, "order": 1},
                    {"section": "วันเวลาการประชุม", "required": True, "order": 2},
                    {"section": "สถานที่", "required": True, "order": 3},
                    {"section": "วาระการประชุม", "required": True, "order": 4},
                    {"section": "ผู้เข้าร่วม", "required": True, "order": 5},
                    {"section": "ลงนาม", "required": True, "order": 6}
                ],
                "formatting": {
                    "numbering": "thai_numerals",
                    "date_format": "buddhist_calendar",
                    "language_style": "formal_invitation"
                },
                "required_elements": ["วันที่", "เวลา", "สถานที่", "เรื่อง"],
                "keywords": ["เชิญ", "ประชุม", "การประชุม", "นัดหมาย"]
            },
            "รายงาน": {
                "structure": [
                    {"section": "หัวข้อเอกสาร", "required": True, "order": 1},
                    {"section": "ขอบเขตการตรวจสอบ", "required": True, "order": 2},
                    {"section": "วิธีการดำเนินการ", "required": True, "order": 3},
                    {"section": "ผลการตรวจสอบ", "required": True, "order": 4},
                    {"section": "สรุป", "required": True, "order": 5},
                    {"section": "ข้อเสนอแนะ", "required": True, "order": 6},
                    {"section": "ลงนาม", "required": True, "order": 7}
                ],
                "formatting": {
                    "numbering": "thai_numerals",
                    "date_format": "buddhist_calendar",
                    "language_style": "formal_report"
                },
                "required_elements": ["เรื่อง", "วันที่", "สถานที่", "ผู้ตรวจสอบ"],
                "keywords": ["รายงาน", "สรุป", "ผลการ", "การตรวจสอบ"]
            }
        }

    def _load_formatting_rules(self) -> Dict:
        """Load formatting rules for Thai documents"""
        return {
            "thai_numerals": {
                "conversion": {
                    "1": "๑", "2": "๒", "3": "๓", "4": "๔", "5": "๕",
                    "6": "๖", "7": "๗", "8": "๘", "9": "๙", "0": "๐"
                },
                "pattern": r"\d+",
                "required_for": ["section_numbers", "document_numbers", "dates"]
            },
            "buddhist_calendar": {
                "conversion": 543,  # Add to Christian year
                "format": "ดดด ๓๓๓๓ พ.ศ. ๒๕๕๕",
                "months": {
                    "1": "มกราคม", "2": "กุมภาพันธ์", "3": "มีนาคม",
                    "4": "เมษายน", "5": "พฤษภาคม", "6": "มิถุนายน",
                    "7": "กรกฎาคม", "8": "สิงหาคม", "9": "กันยายน",
                    "10": "ตุลาคม", "11": "พฤศจิกายน", "12": "ธันวาคม"
                }
            },
            "formal_language": {
                "openings": ["ตามที่", "อ้างถึง", "ด้วยเรื่อง"],
                "transitions": ["ในการนี้", "เพื่อให้", "ดังนั้น"],
                "closings": ["จึงเรียนมาเพื่อ", "จึงขอ", "ขอแสดงความนับถือ"],
                "honorifics": ["ท่าน", "กระผม", "ข้าพเจ้า"]
            }
        }

    def _load_language_patterns(self) -> Dict:
        """Load Thai language patterns for government documents"""
        return {
            "section_headers": [
                "๑. เรื่องเดิม",
                "๒. เรื่องเพื่อพิจารณา",
                "๓. เรื่องเพื่อทราบ",
                "๔. ข้อเท็จจริง",
                "๕. ข้อสังเกต",
                "๖. ข้อเสนอแนะ"
            ],
            "document_references": [
                "ตามบันทึกสำนักงานฯ",
                "อ้างถึงหนังสือฯ",
                "ตามที่ได้รับมอบหมาย",
                "ตามมติคณะกรรมการ"
            ],
            "formal_expressions": {
                "requests": [
                    "ขอให้พิจารณาอนุมัติ",
                    "ขออนุมัติ",
                    "ขอให้ดำเนินการ",
                    "ขอมอบหมาย"
                ],
                "justifications": [
                    "เพื่อความเรียบร้อย",
                    "เพื่อประโยชน์ในการปฏิบัติงาน",
                    "เพื่อให้เป็นไปตามระเบียบ",
                    "เพื่อให้การดำเนินงานเป็นไปอย่างมีประสิทธิภาพ"
                ]
            }
        }

    def classify_request(self, user_query: str) -> Tuple[str, float]:
        """Classify document type from user query"""
        query_lower = user_query.lower()
        scores = {}

        for doc_type, template in self.document_templates.items():
            keywords = template["keywords"]
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            scores[doc_type] = matches / len(keywords) if keywords else 0

        if not scores or max(scores.values()) == 0:
            return "บันทึกข้อความ", 0.5

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        return best_type, confidence

    def extract_document_info(self, user_query: str) -> Dict:
        """Extract structured information from user query"""
        info = {
            "dates": self._extract_dates(user_query),
            "locations": self._extract_locations(user_query),
            "people": self._extract_people(user_query),
            "technical_terms": self._extract_technical_terms(user_query),
            "document_numbers": self._extract_document_numbers(user_query),
            "action_words": self._extract_action_words(user_query)
        }

        return info

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        date_patterns = [
            r'\d{1,2}\s+(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)\s+\d{4}',
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'วันที่\s+\d{1,2}',
            r'\d{1,2}\s+มกราคม.*?\d{4}'
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)

        return dates

    def _extract_locations(self, text: str) -> List[str]:
        """Extract locations from text"""
        location_patterns = [
            r'จังหวัด\s+(\w+)',
            r'อำเภอ\s+(\w+)',
            r'ตำบล\s+(\w+)',
            r'ณ\s+(\w+(?:\s+\w+)*)',
            r'ห้อง\s+(\w+)',
            r'สถานีวิทยุ\s+(\w+(?:\s+\w+)*)'
        ]

        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.extend(matches)

        return locations

    def _extract_people(self, text: str) -> List[str]:
        """Extract people/positions from text"""
        people_patterns = [
            r'คณะกรรมการ\s+(\w+)',
            r'ผู้\w+\s+(\w+)',
            r'นาย\s+(\w+(?:\s+\w+)*)',
            r'นาง\s+(\w+(?:\s+\w+)*)',
            r'นางสาว\s+(\w+(?:\s+\w+)*)',
            r'(\w+)\s+ผู้อำนวยการ',
            r'(\w+)\s+ประธาน'
        ]

        people = []
        for pattern in people_patterns:
            matches = re.findall(pattern, text)
            people.extend(matches)

        return people

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text"""
        technical_keywords = [
            'แก๊งคอลเซ็นเตอร์', 'cellsite', 'สัญญาณ', '4G', '5G',
            'คลื่นความถี่', 'ITU', 'ชายแดน', 'ตรวจสอบ',
            'มาตรฐาน', 'ระบบ', 'เครือข่าย', 'สถานีวิทยุ',
            'คมนาคม', 'โทรคมนาคม', 'วิทยุกระจายเสียง'
        ]

        found_terms = []
        text_lower = text.lower()

        for term in technical_keywords:
            if term.lower() in text_lower:
                found_terms.append(term)

        return found_terms

    def _extract_document_numbers(self, text: str) -> List[str]:
        """Extract document numbers from text"""
        doc_patterns = [
            r'(?:สทช|สสท|อนท|นม)\s*\d+/\d{4}',
            r'ที่\s+\d+/\d{4}',
            r'เลขที่\s+\d+/\d{4}'
        ]

        doc_numbers = []
        for pattern in doc_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            doc_numbers.extend(matches)

        return doc_numbers

    def _extract_action_words(self, text: str) -> List[str]:
        """Extract action words from text"""
        action_keywords = [
            'ขออนุมัติ', 'เชิญประชุม', 'แจ้ง', 'รายงาน',
            'ตรวจสอบ', 'สั่งการ', 'มอบหมาย', 'แต่งตั้ง',
            'เดินทาง', 'ดำเนินการ', 'พิจารณา', 'อนุมัติ'
        ]

        found_actions = []
        text_lower = text.lower()

        for action in action_keywords:
            if action in text_lower:
                found_actions.append(action)

        return found_actions

    def create_context_template(self, document_type: str, section: str,
                              extracted_info: Dict) -> DocumentContext:
        """Create context template for specific document section"""

        if document_type not in self.document_templates:
            document_type = "บันทึกข้อความ"

        template = self.document_templates[document_type]

        # Determine formatting rules for this section
        formatting_rules = [
            f"document_type:{document_type}",
            f"numbering:{template['formatting']['numbering']}",
            f"date_format:{template['formatting']['date_format']}",
            f"language_style:{template['formatting']['language_style']}"
        ]

        # Content requirements based on section
        content_requirements = self._get_section_requirements(section, extracted_info)

        # Language patterns for this document type
        language_patterns = self._get_language_patterns(document_type, section)

        return DocumentContext(
            document_type=document_type,
            section=section,
            formatting_rules=formatting_rules,
            content_requirements=content_requirements,
            language_patterns=language_patterns
        )

    def _get_section_requirements(self, section: str, extracted_info: Dict) -> Dict:
        """Get content requirements for specific section"""
        requirements = {
            "section_name": section,
            "purpose": "",
            "content_type": "",
            "required_elements": [],
            "optional_elements": []
        }

        section_configs = {
            "เรื่องเดิม": {
                "purpose": "Provide background and reference information",
                "content_type": "reference",
                "required_elements": ["document_reference", "background_info"],
                "optional_elements": ["previous_actions", "legal_basis"]
            },
            "เรื่องเพื่อพิจารณา": {
                "purpose": "Present items requiring decision/approval",
                "content_type": "action",
                "required_elements": ["clear_request", "justification"],
                "optional_elements": ["timeline", "resource_requirements"]
            },
            "เรื่องเพื่อทราบ": {
                "purpose": "Provide information only, no action required",
                "content_type": "information",
                "required_elements": ["key_information"],
                "optional_elements": ["additional_context", "contact_info"]
            },
            "วันเวลาการประชุม": {
                "purpose": "Specify meeting schedule",
                "content_type": "schedule",
                "required_elements": ["date", "time", "duration"],
                "optional_elements": ["meeting_format", "technical_requirements"]
            },
            "สถานที่": {
                "purpose": "Specify location details",
                "content_type": "location",
                "required_elements": ["place_name", "address"],
                "optional_elements": ["directions", "parking_info", "contact_person"]
            }
        }

        if section in section_configs:
            requirements.update(section_configs[section])

        # Add extracted information to requirements
        if extracted_info:
            requirements["extracted_info"] = extracted_info

        return requirements

    def _get_language_patterns(self, document_type: str, section: str) -> List[str]:
        """Get language patterns for document type and section"""
        patterns = []

        # Base patterns for document type
        if document_type == "บันทึกข้อความ":
            patterns.extend([
                "ตามที่...",
                "จึงเรียนมาเพื่อโปรดพิจารณา...",
                "จึงเรียนมาเพื่อทราบ..."
            ])
        elif document_type == "หนังสือเชิญ":
            patterns.extend([
                "ด้วยเรื่อง...",
                "ขอเชิญ...",
                "จึงเรียนมาเพื่อโปรดเข้าร่วม..."
            ])
        elif document_type == "รายงาน":
            patterns.extend([
                "จากการตรวจสอบ...",
                "สรุปผลการ...",
                "ขอเสนอแนะ..."
            ])

        # Section-specific patterns
        section_patterns = {
            "เรื่องเดิม": ["ตามบันทึก...", "อ้างถึง...", "ตามที่ได้รับมอบหมาย..."],
            "เรื่องเพื่อพิจารณา": ["ขอให้พิจารณา...", "ขออนุมัติ...", "เพื่อให้..."],
            "เรื่องเพื่อทราบ": ["ขอให้ทราบว่า...", "สำหรับทราบ...", "เพื่อทราบ..."]
        }

        if section in section_patterns:
            patterns.extend(section_patterns[section])

        return patterns

    def generate_context_prompt(self, context: DocumentContext) -> str:
        """Generate optimized context prompt for local model"""

        prompt_parts = [
            f"เป็นทางการ",
            f"เอกสารประเภท: {context.document_type}",
            f"ส่วน: {context.section}",
            f"เป้าหมาย: {context.content_requirements.get('purpose', '')}",
            f"ประเภทเนื้อหา: {context.content_requirements.get('content_type', '')}",
            "",
            "กฎการจัดรูปแบบ:",
            *[f"  - {rule}" for rule in context.formatting_rules],
            "",
            "รูปแบบภาษา:",
            *[f"  - {pattern}" for pattern in context.language_patterns],
            "",
            "ข้อมูลที่ต้องรวม:",
            f"  - จุดประสงค์: {context.content_requirements.get('purpose', '')}",
            f"  - ประเภท: {context.content_requirements.get('content_type', '')}",
        ]

        # Add required elements
        required_elements = context.content_requirements.get('required_elements', [])
        if required_elements:
            prompt_parts.extend([
                "  - องค์ประกอบที่จำเป็น:",
                *[f"    • {element}" for element in required_elements]
            ])

        # Add extracted information if available
        extracted_info = context.content_requirements.get('extracted_info', {})
        if extracted_info:
            prompt_parts.append("")
            prompt_parts.append("ข้อมูลที่สกัดจากคำขอ:")
            for key, value in extracted_info.items():
                if value:
                    prompt_parts.append(f"  - {key}: {', '.join(value) if isinstance(value, list) else value}")

        # Add content generation instructions
        prompt_parts.extend([
            "",
            "คำแนะนำในการสร้างเนื้อหา:",
            f"  ๑. เขียนเนื้อหาสำหรับส่วน '{context.section}' เท่านั้น",
            f"  ๒. ใช้ภาษาราชการที่เป็นทางการและสุภาพ",
            f"  ๓. ใช้เลขไทย (๑, ๒, ๓) สำหรับการลำดับข้อ",
            f"  ๔. ต้องรวมข้อมูลจากคำขอต้นฉบับทั้งหมด",
            f"  ๕. จัดระเบียบเนื้อหาตามโครงสร้างที่กำหนด"
        ])

        return "\n".join(prompt_parts)

    def optimize_context_for_local_model(self, prompt: str, max_tokens: int = 2048) -> str:
        """Optimize context for local model with token constraints"""

        # Simple token estimation (1 token ≈ 4 characters for Thai)
        estimated_tokens = len(prompt) // 4

        if estimated_tokens <= max_tokens:
            return prompt

        # If too long, prioritize key information
        key_sections = [
            "เอกสารประเภท:",
            "กฎการจัดรูปแบบ:",
            "ข้อมูลที่ต้องรวม:",
            "คำแนะนำในการสร้างเนื้อหา:"
        ]

        optimized_parts = []
        lines = prompt.split('\n')
        current_section = None

        for line in lines:
            # Check if this is a key section header
            for section in key_sections:
                if line.startswith(section):
                    current_section = section
                    optimized_parts.append(line)
                    break
            else:
                # Include line if it's part of a key section
                if current_section and line.strip():
                    optimized_parts.append(line)

        return "\n".join(optimized_parts)

# Factory function
def create_context_engineer() -> ThaiDocumentContextEngineer:
    """Create context engineer instance"""
    return ThaiDocumentContextEngineer()