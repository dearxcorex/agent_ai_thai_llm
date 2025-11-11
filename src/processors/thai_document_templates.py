#!/usr/bin/env python3
"""
Thai Document Templates and Structure Patterns
Provides document-type-specific templates for Thai official documents
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import re

@dataclass
class DocumentTemplate:
    """Template structure for Thai official documents"""
    document_type: str
    structure: List[str]
    required_sections: List[str]
    formatting_rules: List[str]
    example_patterns: List[str]

class ThaiDocumentTemplates:
    """Predefined templates for Thai government document types"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, DocumentTemplate]:
        """Initialize document type templates"""
        templates = {}

        # หนังสือเชิญ (Invitation Letter)
        templates["หนังสือเชิญ"] = DocumentTemplate(
            document_type="หนังสือเชิญ",
            structure=[
                "วันที่",
                "เรื่อง",
                "เรียน",
                "อ้างถึง (ถ้ามี)",
                "เนื้อหาการเชิญ",
                "วันเวลาและสถานที่",
                "จึงเรียนมาเพื่อโปรดพิจารณา",
                "ขอแสดงความนับถือ"
            ],
            required_sections=["เรื่อง", "เรียน", "วันเวลา", "สถานที่"],
            formatting_rules=[
                "ใช้เลขไทยในการลำดับข้อ",
                "ระบุวันที่เป็นพุทธศักราช",
                "มีคำว่า 'เรียน' ตามด้วยชื่อผู้รับสาส์น",
                "ระบุวันเวลาและสถานที่ชัดเจน"
            ],
            example_patterns=[
                "เรื่อง เรียนเชิญประชุม",
                "ในวันที่... เวลา... น. ณ...",
                "จึงเรียนมาเพื่อโปรดพิจารณา"
            ]
        )

        # รายงาน (Report)
        templates["รายงาน"] = DocumentTemplate(
            document_type="รายงาน",
            structure=[
                "ชื่อรายงาน",
                "วันที่จัดทำ",
                "ผู้จัดทำ/หน่วยงาน",
                "เนื้อหารายงาน",
                "สรุปผลการดำเนินการ",
                "ข้อเสนอแนะ (ถ้ามี)",
                "ลงชื่อผู้จัดทำ"
            ],
            required_sections=["ชื่อรายงาน", "วันที่", "เนื้อหา", "สรุป"],
            formatting_rules=[
                "มีหัวข้อรายงานที่ชัดเจน",
                "ระบุวันที่จัดทำเป็นพุทธศักราช",
                "แบ่งเนื้อหาเป็นส่วนๆ ชัดเจน",
                "มีสรุปผลการดำเนินการ"
            ],
            example_patterns=[
                "รายงานการตรวจสอบ...",
                "สรุปผลการดำเนินการ",
                "จึงรายงานให้ทราบ"
            ]
        )

        # หนังสือร้องขอ (Request Letter)
        templates["หนังสือร้องขอ"] = DocumentTemplate(
            document_type="หนังสือร้องขอ",
            structure=[
                "วันที่",
                "เรื่อง",
                "เรียน",
                "อ้างถึง (ถ้ามี)",
                "เหตุผลและความจำเป็น",
                "รายละเอียดการร้องขอ",
                "จึงเรียนมาเพื่อโปรดพิจารณา/อนุมัติ",
                "ขอแสดงความนับถือ"
            ],
            required_sections=["เรื่อง", "เรียน", "เหตุผล", "รายละเอียด"],
            formatting_rules=[
                "ระบุเหตุผลที่ชัดเจน",
                "มีรายละเอียนการร้องขอครบถ้วน",
                "ใช้ภาษาราชการที่สุภาพ",
                "ระบุวัตถุประสงค์โดยชัดเจน"
            ],
            example_patterns=[
                "เรื่อง ขอร้องขอ...",
                "ด้วยความจำเป็น...",
                "จึงเรียนมาเพื่อโปรดพิจารณาอนุมัติ"
            ]
        )

        # บันทึกข้อความ (Internal Memo)
        templates["บันทึกข้อความ"] = DocumentTemplate(
            document_type="บันทึกข้อความ",
            structure=[
                "วันที่",
                "เรื่อง",
                "เรียน",
                "สาระสำคัญ",
                "ข้อความ",
                "ข้อเสนอแนะ (ถ้ามี)",
                "เพื่อพิจารณา/ทราบ",
                "ลงชื่อและตำแหน่ง"
            ],
            required_sections=["วันที่", "เรื่อง", "เรียน", "ข้อความ"],
            formatting_rules=[
                "กระชับและชัดเจน",
                "ใช้ภาษาที่เป็นทางการ",
                "มีวัตถุประสงค์ที่ชัดเจน",
                "แยกส่วนเพื่อพิจารณาและเพื่อทราบ"
            ],
            example_patterns=[
                "เรื่อง เพื่อโปรดทราบ",
                "สาระสำคัญ",
                "เพื่อพิจารณาและเพื่อทราบ"
            ]
        )

        # หนังสืออนุมัติ (Approval Letter)
        templates["หนังสืออนุมัติ"] = DocumentTemplate(
            document_type="หนังสืออนุมัติ",
            structure=[
                "วันที่",
                "เรื่อง",
                "เรียน",
                "อ้างถึงหนังสือร้องขอ",
                "มติคณะกรรมการ/การพิจารณา",
                "เงื่อนไขการอนุมัติ (ถ้ามี)",
                "ดำเนินการตามนี้",
                "ขอแสดงความนับถือ"
            ],
            required_sections=["เรื่อง", "เรียน", "อ้างถึง", "การอนุมัติ"],
            formatting_rules=[
                "อ้างถึงหนังสือร้องขอเดิม",
                "ระบุเงื่อนไขการอนุมัติชัดเจน",
                "มีมติคณะกรรมการ (ถ้าใช้)",
                "ระบุขั้นตอนการดำเนินการ"
            ],
            example_patterns=[
                "ตามหนังสือร้องขอเรื่อง...",
                "คณะกรรมการมีมติอนุมัติ",
                "ให้ดำเนินการตามนี้"
            ]
        )

        return templates

    def get_template(self, document_type: str) -> Optional[DocumentTemplate]:
        """Get template by document type"""
        return self.templates.get(document_type)

    def get_available_types(self) -> List[str]:
        """Get list of available document types"""
        return list(self.templates.keys())

    def validate_structure(self, document_type: str, content: str) -> Dict:
        """Validate document structure against template"""
        template = self.get_template(document_type)
        if not template:
            return {
                "is_valid": False,
                "missing_sections": [],
                "score": 0.0,
                "reason": f"No template found for document type: {document_type}"
            }

        content_lower = content.lower()
        missing_sections = []
        found_sections = []

        for section in template.required_sections:
            if section in content_lower:
                found_sections.append(section)
            else:
                missing_sections.append(section)

        score = len(found_sections) / len(template.required_sections)
        is_valid = score >= 0.7  # 70% of required sections

        return {
            "is_valid": is_valid,
            "missing_sections": missing_sections,
            "found_sections": found_sections,
            "score": score,
            "template_used": template.document_type
        }

    def generate_template_prompt(self, document_type: str) -> str:
        """Generate formatting prompt based on document type template"""
        template = self.get_template(document_type)
        if not template:
            return ""

        prompt = f"""
โครงสร้างเอกสารประเภท {document_type}:

ส่วนที่ต้องรวม:
{chr(10).join([f"- {section}" for section in template.required_sections])}

กฎการจัดรูปแบบ:
{chr(10).join([f"- {rule}" for rule in template.formatting_rules])}

ลำดับโครงสร้าง:
{chr(10).join([f"{i+1}. {section}" for i, section in enumerate(template.structure)])}

ตัวอย่างประโยค:
{chr(10).join([f"- {pattern}" for pattern in template.example_patterns])}
"""

        return prompt.strip()

class ThaiDocumentStructureAnalyzer:
    """Analyze and classify Thai document structures"""

    def __init__(self):
        self.templates = ThaiDocumentTemplates()

    def classify_document_type(self, content: str) -> Dict:
        """Classify document type based on content structure"""
        content_lower = content.lower()

        scores = {}
        for doc_type in self.templates.get_available_types():
            validation = self.templates.validate_structure(doc_type, content_lower)
            scores[doc_type] = validation["score"]

        # Get best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Sort all scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "predicted_type": best_type if best_score >= 0.5 else "บันทึกข้อความ",
            "confidence": best_score,
            "all_scores": sorted_scores[:3],  # Top 3 matches
            "template_match": self.templates.get_template(best_type) is not None
        }

    def extract_document_components(self, content: str, document_type: str) -> Dict:
        """Extract structured components from document"""
        template = self.templates.get_template(document_type)
        if not template:
            return {"components": {}, "template_found": False}

        components = {}
        content_lower = content.lower()

        # Extract common components
        patterns = {
            "วันที่": r"วันที่\s*(.+?)(?=\n|$)",
            "เรื่อง": r"เรื่อง\s*(.+?)(?=\n|$)",
            "เรียน": r"เรียน\s*(.+?)(?=\n|$)",
            "อ้างถึง": r"อ้างถึง\s*(.+?)(?=\n|$)",
            "สถานที่": r"ณ\s*(.+?)(?=\n|วันที่|เวลา)",
            "เวลา": r"เวลา\s*(.+?)(?=\n|น\.|ณ)",
        }

        for component, pattern in patterns.items():
            matches = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                components[component] = matches.group(1).strip()

        return {
            "components": components,
            "template_found": True,
            "document_type": document_type
        }

# Factory function
def create_thai_document_templates() -> ThaiDocumentTemplates:
    """Create Thai document templates instance"""
    return ThaiDocumentTemplates()

def create_document_analyzer() -> ThaiDocumentStructureAnalyzer:
    """Create document structure analyzer"""
    return ThaiDocumentStructureAnalyzer()

# Test function
if __name__ == "__main__":
    templates = ThaiDocumentTemplates()
    analyzer = ThaiDocumentStructureAnalyzer()

    # Test document
    test_content = """
    เรื่อง เรียนเชิญประชุม
    เรียน ประธานคณะกรรมการ
    ในวันที่ ๑๐ กรกฎาคม ๒๕๖๘ เวลา ๐๙:๐๐ น. ณ ห้องประชุมใหญ่
    จึงเรียนมาเพื่อโปรดพิจารณา
    """

    # Test classification
    result = analyzer.classify_document_type(test_content)
    print(f"Classification result: {result}")

    # Test template validation
    template = templates.get_template("หนังสือเชิญ")
    if template:
        print(f"Template found: {template.document_type}")
        print(f"Required sections: {template.required_sections}")