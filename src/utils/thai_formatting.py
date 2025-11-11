"""
Thai Formatting Utilities for Government Documents
Provides Thai numeral conversion, date formatting, and official language patterns
"""

import re
from datetime import datetime
from typing import Dict, List, Optional

class ThaiFormatter:
    """Thai government document formatting utilities"""

    def __init__(self):
        self.thai_numerals = {
            "0": "๐", "1": "๑", "2": "๒", "3": "๓", "4": "๔",
            "5": "๕", "6": "๖", "7": "๗", "8": "๘", "9": "๙"
        }

        self.thai_months = {
            1: "มกราคม", 2: "กุมภาพันธ์", 3: "มีนาคม",
            4: "เมษายน", 5: "พฤษภาคม", 6: "มิถุนายน",
            7: "กรกฎาคม", 8: "สิงหาคม", 9: "กันยายน",
            10: "ตุลาคม", 11: "พฤศจิกายน", 12: "ธันวาคม"
        }

    def convert_to_thai_numbers(self, text: str) -> str:
        """Convert Arabic numerals to Thai numerals"""
        result = text
        for arabic, thai in self.thai_numerals.items():
            result = result.replace(arabic, thai)
        return result

    def format_thai_date(self, date_str: str, to_buddhist: bool = True) -> str:
        """
        Format date string to Thai format

        Args:
            date_str: Date string in various formats
            to_buddhist: Convert to Buddhist calendar

        Returns:
            Formatted Thai date string
        """
        try:
            # Try to parse the date
            if isinstance(date_str, str):
                # Try different date formats
                for fmt in ["%d %B %Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"]:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If parsing fails, return original string with Thai numerals
                    return self.convert_to_thai_numbers(date_str)
            else:
                date_obj = date_str

            # Format to Thai
            day = self.convert_to_thai_numbers(f"{date_obj.day:02d}")
            month = self.thai_months[date_obj.month]
            year = date_obj.year

            if to_buddhist:
                year += 543

            year = self.convert_to_thai_numbers(f"{year:04d}")

            return f"{day} {month} {year}"

        except Exception:
            # If all else fails, just convert numerals
            return self.convert_to_thai_numbers(str(date_str))

    def format_document_number(self, doc_number: str) -> str:
        """Format document number with Thai numerals"""
        # Extract numbers and convert to Thai
        return re.sub(r'\d+', lambda m: self.convert_to_thai_numbers(m.group()), doc_number)

    def apply_thai_formatting(self, text: str) -> str:
        """Apply comprehensive Thai formatting to text"""

        # Convert section numbers to Thai numerals
        text = re.sub(r'^(\d+)\.', lambda m: self.convert_to_thai_numbers(m.group(1)) + '.', text, flags=re.MULTILINE)

        # Convert numbered lists to Thai numerals
        text = re.sub(r'^(\d+)\)', lambda m: self.convert_to_thai_numbers(m.group(1)) + ')', text, flags=re.MULTILINE)

        # Convert document numbers
        text = re.sub(r'(สทช|สสท|อนท|นม)\s*(\d+)/(\d{4})',
                     lambda m: f"{m.group(1)} {self.convert_to_thai_numbers(m.group(2))}/{self.convert_to_thai_numbers(m.group(3))}",
                     text)

        # Convert standalone numbers in document context
        text = re.sub(r'\b(\d{1,2})\s+(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)\s+(\d{4})',
                     lambda m: f"{self.convert_to_thai_numbers(m.group(1))} {m.group(2)} {self.convert_to_thai_numbers(str(int(m.group(3)) + 543))}",
                     text)

        # Convert time format
        text = re.sub(r'(\d{1,2}):(\d{2})\s*น\.?',
                     lambda m: f"{self.convert_to_thai_numbers(m.group(1))}.{self.convert_to_thai_numbers(m.group(2))} น.",
                     text)

        return text

    def validate_thai_formatting(self, text: str) -> Dict:
        """Validate Thai formatting compliance"""

        issues = []
        is_compliant = True

        # Check for Thai numerals in sections
        if re.search(r'^\d+\.', text, re.MULTILINE):
            issues.append("Found Arabic numerals in section numbering")
            is_compliant = False

        # Check for Buddhist calendar years
        if re.search(r'\b(25[0-4]\d|25[5-9]\d|2[6-9]\d\d)\b', text):
            # Has Buddhist years - good
            pass
        elif re.search(r'\b(19|20)\d{2}\b', text):
            # Has Christian years - issue
            issues.append("Found Christian calendar years instead of Buddhist")
            is_compliant = False

        # Check for formal language patterns
        formal_patterns = ["ตามที่", "จึงเรียนมาเพื่อ", "ขอแสดงความนับถือ"]
        has_formal_language = any(pattern in text for pattern in formal_patterns)

        if not has_formal_language:
            issues.append("Missing formal Thai government language patterns")
            is_compliant = False

        return {
            "is_compliant": is_compliant,
            "issues": issues,
            "compliance_score": 1.0 - (len(issues) * 0.2)
        }

class ThaiGovernmentTerminology:
    """Thai government official terminology and expressions"""

    FORMAL_OPENINGS = [
        "ตามที่ได้รับมอบหมาย",
        "ตามบันทึกสำนักงานฯ",
        "อ้างถึงหนังสือฯ",
        "ด้วยเรื่อง",
        "ในการนี้"
    ]

    FORMAL_CLOSINGS = [
        "จึงเรียนมาเพื่อโปรดพิจารณา",
        "จึงเรียนมาเพื่อโปรดอนุมัติ",
        "จึงเรียนมาเพื่อทราบ",
        "จึงขอแสดงความนับถือ",
        "ขอแสดงความนับถือ"
    ]

    REQUEST_EXPRESSIONS = [
        "ขอให้พิจารณาอนุมัติ",
        "ขออนุมัติ",
        "ขอให้ดำเนินการ",
        "ขอมอบหมาย",
        "ขอเชิญประชุม",
        "ขอรับทราบ"
    ]

    REFERENCE_EXPRESSIONS = [
        "ตามมติคณะกรรมการ",
        "ตามระเบียบสำนักงานฯ",
        "ตามกฎหมายว่าด้วย",
        "ตามนโยบายของ",
        "ตามที่ได้กำหนดไว้"
    ]

    TECHNICAL_TERMS = {
        "telecommunications": [
            "สถานีวิทยุคมนาคม",
            "คลื่นความถี่",
            "โทรคมนาคม",
            "วิทยุกระจายเสียง",
            "วิทยุโทรทัศน์",
            "โทรศัพท์",
            "โทรคมนาคมผ่านดาวเทียม"
        ],
        "regulatory": [
            "กสทช",
            "สำนักงาน กสทช",
            "คณะกรรมการกิจการกระจายเสียง",
            "คณะกรรมการกิจการโทรทัศน์",
            "คณะกรรมการกิจการโทรคมนาคม",
            "มติคณะกรรมการ",
            "ระเบียบสำนักงานฯ"
        ],
        "legal": [
            "พระราชบัญญัติ",
            "กฎหมาย",
            "ระเบียบ",
            "ข้อบังคับ",
            "ประกาศ",
            "มาตรา",
            "ข้อกำหนด"
        ]
    }

    HONORIFICS = {
        "director": "ผู้อำนวยการ",
        "chairman": "ประธาน",
        "secretary": "เลขานุการ",
        "committee": "คณะกรรมการ",
        "official": "ข้าราชการ",
        "engineer": "วิศวกร",
        "analyst": "นักวิเคราะห์",
        "inspector": "ผู้ตรวจสอบ"
    }

    @classmethod
    def get_formal_opening(cls, context_type: str = "general") -> str:
        """Get appropriate formal opening based on context"""
        openings_map = {
            "reference": cls.FORMAL_OPENINGS[:3],
            "new_matter": cls.FORMAL_OPENINGS[3:],
            "general": cls.FORMAL_OPENINGS
        }
        return opennings_map.get(context_type, cls.FORMAL_OPENINGS)[0]

    @classmethod
    def get_formal_closing(cls, request_type: str = "general") -> str:
        """Get appropriate formal closing based on request type"""
        closing_map = {
            "approval": cls.FORMAL_CLOSINGS[:2],
            "information": cls.FORMAL_CLOSINGS[2:3],
            "respect": cls.FORMAL_CLOSINGS[3:],
            "general": cls.FORMAL_CLOSINGS
        }
        return closing_map.get(request_type, cls.FORMAL_CLOSINGS)[0]

    @classmethod
    def get_technical_terms(cls, domain: str) -> List[str]:
        """Get technical terms for specific domain"""
        return cls.TECHNICAL_TERMS.get(domain, [])

    @classmethod
    def validate_formal_language(cls, text: str) -> Dict:
        """Validate formal Thai government language usage"""

        issues = []
        is_formal = True

        # Check for formal openings
        has_formal_opening = any(opening in text for opening in cls.FORMAL_OPENINGS)
        if not has_formal_opening:
            issues.append("Missing formal opening expression")
            is_formal = False

        # Check for formal closings
        has_formal_closing = any(closing in text for closing in cls.FORMAL_CLOSINGS)
        if not has_formal_closing:
            issues.append("Missing formal closing expression")
            is_formal = False

        # Check for appropriate technical terms
        tech_domains = ["telecommunications", "regulatory", "legal"]
        has_tech_terms = False
        for domain in tech_domains:
            if any(term in text for term in cls.TECHNICAL_TERMS.get(domain, [])):
                has_tech_terms = True
                break

        if not has_tech_terms:
            issues.append("Missing appropriate technical terminology")
            is_formal = False

        return {
            "is_formal": is_formal,
            "issues": issues,
            "formality_score": 1.0 - (len(issues) * 0.15)
        }

def validate_thai_document(document: str) -> Dict:
    """Comprehensive Thai government document validation"""

    formatter = ThaiFormatter()
    terminology = ThaiGovernmentTerminology()

    # Validate formatting
    format_result = formatter.validate_thai_formatting(document)

    # Validate formal language
    language_result = terminology.validate_formal_language(document)

    # Calculate overall score
    overall_score = (format_result["compliance_score"] + language_result["formality_score"]) / 2

    all_issues = format_result["issues"] + language_result["issues"]

    return {
        "overall_score": overall_score,
        "is_compliant": overall_score >= 0.7,
        "formatting": format_result,
        "language": language_result,
        "issues": all_issues,
        "recommendations": [
            "Apply proper Thai numeral formatting",
            "Use formal government language patterns",
            "Include appropriate technical terminology",
            "Follow official document structure"
        ]
    }

def format_thai_document(text: str) -> str:
    """Apply comprehensive Thai formatting to document text"""
    formatter = ThaiFormatter()
    return formatter.apply_thai_formatting(text)