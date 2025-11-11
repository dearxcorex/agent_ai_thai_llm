#!/usr/bin/env python3
"""
Test script for Context Engineered RAG System
Demonstrates enhanced accuracy with DeepSeek and context engineering
"""

import sys
import logging
from datetime import datetime
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_context_engineered_test():
    """Run comprehensive test of context engineered RAG system"""

    logger.info("🚀 Starting Context Engineered RAG Test")
    logger.info("=" * 60)

    try:
        # Import the system
        from context_engineered_rag import app

        # Test cases covering different document types
        test_cases = [
            {
                "id": 1,
                "name": "Travel Approval Memo",
                "query": "เขียนหนังสือบันทึกข้อความ ขออนุมัติเดินทางไปตรวจสอบสถานีวิทยุคมนาคม จังหวัดบุรีรัมย์ 1-2 มกราคม 2568 ตามแผนปฏิบัติการ",
                "expected_type": "บันทึกข้อความ",
                "focus_areas": ["section_separation", "thai_format", "travel_details"]
            },
            {
                "id": 2,
                "name": "ITU Meeting Invitation",
                "query": "สร้างหนังสือเชิญประชุมกับคณะกรรมการ ITU เรื่องการป้องกันแก๊ง call center ในประเทศเพื่อนบ้าน โดยจะเสนอมาตรฐานการลดกำลังส่ง cellsite เพื่อไม่ให้ใช้สัญญาณข้ามพรมแดน กำหนดวันที่ 10 กรกฎาคม 2568 เวลา 09:00 น. ณ ห้องประชุมใหญ่ สำนักงาน กสทช",
                "expected_type": "หนังสือเชิญ",
                "focus_areas": ["meeting_details", "technical_content", "formal_invitation"]
            },
            {
                "id": 3,
                "name": "Inspection Report",
                "query": "จัดทำรายงานการตรวจสอบสถานีวิทยุคมนาคมประจำปี 2568 โดยมีรายละเอียด ตรวจสอบคลื่นความถี่ มาตรฐานการใช้งาน และการปฏิบัติตามกฎหมาย",
                "expected_type": "รายงาน",
                "focus_areas": ["technical_specifications", "compliance", "formal_reporting"]
            }
        ]

        results = []

        for test_case in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST CASE {test_case['id']}: {test_case['name']}")
            logger.info(f"Expected Type: {test_case['expected_type']}")
            logger.info(f"Query: {test_case['query'][:100]}...")
            logger.info(f"{'='*60}")

            # Prepare input state
            input_state = {
                "question": test_case["query"],
                "reflection_iteration": 0,
                "web_search_needed": False,
                "query_rewrite_needed": False,
                "regeneration_needed": False,
                "warnings": []
            }

            # Configure execution
            run_config = {
                "configurable": {"thread_id": f"test_{test_case['id']}"},
                "recursion_limit": 50
            }

            # Execute test
            start_time = datetime.now()
            final_state = None
            last_generation = None

            for output in app.stream(input_state, run_config):
                for node_name, node_output in output.items():
                    if node_name == "analyze_context":
                        logger.info(f"📋 Document Type: {node_output.get('document_type')}")
                        logger.info(f"🔍 Extracted {len(node_output.get('extracted_info', {}))} info categories")
                    elif node_name == "grade_documents":
                        logger.info(f"📊 Average Confidence: {node_output.get('average_confidence', 0):.2f}")
                        logger.info(f"📄 Filtered Documents: {len(node_output.get('filtered_docs', []))}")
                    elif node_name == "generate":
                        generation_length = len(node_output.get('generation', ''))
                        logger.info(f"✍️ Generated Document ({generation_length} chars)")
                    elif node_name == "reflect":
                        quality_score = node_output.get('quality_score', 0)
                        logger.info(f"🔍 Quality Score: {quality_score:.2f}")

                    if 'generation' in node_output:
                        last_generation = node_output['generation']
                    final_state = node_output

            execution_time = (datetime.now() - start_time).total_seconds()

            # Analyze results
            result = analyze_test_results(test_case, final_state, last_generation, execution_time)
            results.append(result)

            # Print test summary
            print_test_summary(result)

        # Print overall summary
        print_overall_summary(results)

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return False

    return True

def analyze_test_results(test_case: Dict, final_state: Dict, generation: str, execution_time: float) -> Dict:
    """Analyze test results and create summary"""

    result = {
        "test_case": test_case,
        "execution_time": execution_time,
        "success": False,
        "document_type_match": False,
        "quality_metrics": {},
        "context_analysis": {},
        "thai_format_compliance": {},
        "section_separation_quality": 0,
        "issues": []
    }

    try:
        # Basic success check
        if final_state and generation and not final_state.get("error"):
            result["success"] = True

        # Document type matching
        actual_type = final_state.get("document_type", "") if final_state else ""
        expected_type = test_case.get("expected_type", "")
        result["document_type_match"] = actual_type == expected_type
        result["context_analysis"] = {
            "actual_type": actual_type,
            "expected_type": expected_type,
            "type_confidence": final_state.get("average_confidence", 0) if final_state else 0
        }

        # Quality metrics
        if final_state:
            quality_assessment = final_state.get("quality_assessment", {})
            thai_validation = final_state.get("thai_format_validation", {})

            result["quality_metrics"] = {
                "overall_score": final_state.get("quality_score", 0),
                "format_correctness": quality_assessment.get("format_correctness", 0),
                "content_accuracy": quality_assessment.get("content_accuracy", 0),
                "thai_format_compliance": quality_assessment.get("thai_format_compliance", 0),
                "section_separation": quality_assessment.get("section_separation", 0)
            }

            result["thai_format_compliance"] = {
                "compliance_score": thai_validation.get("overall_score", 0),
                "is_compliant": thai_validation.get("is_compliant", False),
                "issues": thai_validation.get("issues", [])
            }

            result["section_separation_quality"] = quality_assessment.get("section_separation", 0)

        # Content analysis
        if generation:
            content_analysis = analyze_content_quality(generation, test_case["focus_areas"])
            result["content_analysis"] = content_analysis

            # Check for specific issues
            if content_analysis.get("has_duplicate_sections"):
                result["issues"].append("Duplicate content between sections 2 and 3")
            if not content_analysis.get("has_thai_numerals"):
                result["issues"].append("Missing Thai numerals")
            if not content_analysis.get("has_proper_sections"):
                result["issues"].append("Improper section structure")

    except Exception as e:
        logger.error(f"Result analysis failed: {e}")
        result["issues"].append(f"Analysis error: {str(e)}")

    return result

def analyze_content_quality(document: str, focus_areas: List[str]) -> Dict:
    """Analyze document content quality"""

    analysis = {
        "has_thai_numerals": bool(re.search(r'[๑-๙]', document)),
        "has_proper_sections": False,
        "has_duplicate_sections": False,
        "section_count": 0,
        "word_count": len(document.split()),
        "line_count": len(document.split('\n')),
        "focus_area_coverage": {}
    }

    # Check for proper sections
    section_patterns = [
        r"๑\.\s*เรื่องเดิม",
        r"๒\.\s*เรื่องเพื่อพิจารณา",
        r"๓\.\s*เรื่องเพื่อทราบ"
    ]

    sections_found = []
    for pattern in section_patterns:
        if re.search(pattern, document):
            sections_found.append(pattern)

    analysis["has_proper_sections"] = len(sections_found) >= 2
    analysis["section_count"] = len(sections_found)

    # Check for section separation quality
    if "๒\.\s*เรื่องเพื่อพิจารณา" in document and "๓\.\s*เรื่องเพื่อทราบ" in document:
        # Simple check for content overlap
        section_2_match = re.search(r'๒\.\s*เรื่องเพื่อพิจารณา.*?(?=๓\.\s*เรื่องเพื่อทราบ|$)', document, re.DOTALL)
        section_3_match = re.search(r'๓\.\s*เรื่องเพื่อทราบ.*?(?=จึงเรียนมาเพื่อ|จึงขอ|$)', document, re.DOTALL)

        if section_2_match and section_3_match:
            section_2_text = section_2_match.group().strip()
            section_3_text = section_3_match.group().strip()

            # Simple similarity check
            words_2 = set(section_2_text.lower().split())
            words_3 = set(section_3_text.lower().split())

            if words_2 and words_3:
                intersection = len(words_2.intersection(words_3))
                union = len(words_2.union(words_3))
                similarity = intersection / union if union > 0 else 0

                analysis["section_separation_quality"] = 1 - similarity  # Higher is better
                analysis["has_duplicate_sections"] = similarity > 0.5

    # Check focus area coverage
    for area in focus_areas:
        if area == "meeting_details":
            has_meeting_info = any(term in document for term in ["วันที่", "เวลา", "สถานที่", "ห้องประชุม"])
            analysis["focus_area_coverage"][area] = has_meeting_info
        elif area == "travel_details":
            has_travel_info = any(term in document for term in ["เดินทาง", "จังหวัด", "พนักงาน"])
            analysis["focus_area_coverage"][area] = has_travel_info
        elif area == "technical_content":
            has_tech_info = any(term in document for term in ["cellsite", "สัญญาณ", "คลื่นความถี่", "มาตรฐาน"])
            analysis["focus_area_coverage"][area] = has_tech_info
        elif area == "thai_format":
            analysis["focus_area_coverage"][area] = analysis["has_thai_numerals"]
        elif area == "section_separation":
            analysis["focus_area_coverage"][area] = analysis["has_proper_sections"] and not analysis["has_duplicate_sections"]

    return analysis

def print_test_summary(result: Dict):
    """Print individual test summary"""

    print(f"\n📊 TEST CASE {result['test_case']['id']} SUMMARY")
    print("=" * 50)

    test_case = result["test_case"]
    print(f"📝 Test: {test_case['name']}")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f} seconds")
    print(f"✅ Success: {'Yes' if result['success'] else 'No'}")

    # Context Analysis
    context = result.get("context_analysis", {})
    print(f"\n📋 Context Engineering:")
    print(f"   Document Type: {context.get('actual_type', 'Unknown')}")
    print(f"   Type Match: {'✅' if result['document_type_match'] else '❌'}")
    print(f"   Confidence: {context.get('type_confidence', 0):.2f}")

    # Quality Metrics
    quality = result.get("quality_metrics", {})
    if quality:
        print(f"\n📈 Quality Metrics:")
        print(f"   Overall Score: {quality.get('overall_score', 0):.2f}")
        print(f"   Thai Format: {quality.get('thai_format_compliance', 0):.2f}")
        print(f"   Section Separation: {quality.get('section_separation', 0):.2f}")
        print(f"   Content Accuracy: {quality.get('content_accuracy', 0):.2f}")

    # Thai Format Compliance
    thai_format = result.get("thai_format_compliance", {})
    if thai_format:
        print(f"\n🇹🇭 Thai Format Compliance:")
        print(f"   Compliance: {'✅' if thai_format.get('is_compliant') else '❌'}")
        print(f"   Score: {thai_format.get('compliance_score', 0):.2f}")
        if thai_format.get('issues'):
            print(f"   Issues: {', '.join(thai_format['issues'])}")

    # Content Analysis
    content = result.get("content_analysis", {})
    if content:
        print(f"\n📄 Content Analysis:")
        print(f"   Thai Numerals: {'✅' if content.get('has_thai_numerals') else '❌'}")
        print(f"   Proper Sections: {'✅' if content.get('has_proper_sections') else '❌'}")
        print(f"   No Duplicate Sections: {'✅' if not content.get('has_duplicate_sections') else '❌'}")
        print(f"   Word Count: {content.get('word_count', 0)}")

        coverage = content.get("focus_area_coverage", {})
        if coverage:
            print(f"   Focus Area Coverage:")
            for area, covered in coverage.items():
                status = "✅" if covered else "❌"
                print(f"     {area}: {status}")

    # Issues
    if result.get("issues"):
        print(f"\n⚠️  Issues:")
        for issue in result["issues"]:
            print(f"   • {issue}")

def print_overall_summary(results: List[Dict]):
    """Print overall test summary"""

    print(f"\n{'='*80}")
    print("📊 OVERALL TEST SUMMARY")
    print(f"{'='*80}")

    if not results:
        print("❌ No test results available")
        return

    # Success rates
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    type_matches = sum(1 for r in results if r["document_type_match"])

    print(f"\n🎯 Success Rates:")
    print(f"   Overall Success: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    print(f"   Document Type Accuracy: {type_matches}/{total_tests} ({type_matches/total_tests*100:.1f}%)")

    # Quality averages
    avg_quality = sum(r.get("quality_metrics", {}).get("overall_score", 0) for r in results) / total_tests
    avg_thai_format = sum(r.get("quality_metrics", {}).get("thai_format_compliance", 0) for r in results) / total_tests
    avg_section_separation = sum(r.get("quality_metrics", {}).get("section_separation", 0) for r in results) / total_tests

    print(f"\n📈 Average Quality Metrics:")
    print(f"   Overall Quality: {avg_quality:.2f}")
    print(f"   Thai Format Compliance: {avg_thai_format:.2f}")
    print(f"   Section Separation: {avg_section_separation:.2f}")

    # Performance
    avg_execution_time = sum(r["execution_time"] for r in results) / total_tests
    print(f"\n⏱️  Performance:")
    print(f"   Average Execution Time: {avg_execution_time:.2f} seconds")

    # Content analysis summary
    thai_numeral_tests = sum(1 for r in results if r.get("content_analysis", {}).get("has_thai_numerals"))
    proper_section_tests = sum(1 for r in results if r.get("content_analysis", {}).get("has_proper_sections"))
    no_duplicate_tests = sum(1 for r in results if not r.get("content_analysis", {}).get("has_duplicate_sections"))

    print(f"\n📄 Content Quality:")
    print(f"   Thai Numerals Usage: {thai_numeral_tests}/{total_tests} ({thai_numeral_tests/total_tests*100:.1f}%)")
    print(f"   Proper Section Structure: {proper_section_tests}/{total_tests} ({proper_section_tests/total_tests*100:.1f}%)")
    print(f"   No Duplicate Content: {no_duplicate_tests}/{total_tests} ({no_duplicate_tests/total_tests*100:.1f}%)")

    # Success criteria
    success_criteria = {
        "success_rate": successful_tests/total_tests >= 0.8,
        "type_accuracy": type_matches/total_tests >= 0.8,
        "quality_score": avg_quality >= 0.7,
        "thai_format": avg_thai_format >= 0.8,
        "section_separation": avg_section_separation >= 0.7
    }

    print(f"\n✅ Success Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"   {criterion}: {status}")

    overall_success = all(success_criteria.values())
    print(f"\n🏆 Overall Success: {'✅ PASSED' if overall_success else '❌ FAILED'}")

    # Recommendations
    if not overall_success:
        print(f"\n💡 Recommendations:")
        if not success_criteria["thai_format"]:
            print("   - Improve Thai numeral and date formatting")
        if not success_criteria["section_separation"]:
            print("   - Enhance section separation between 'เพื่อพิจารณา' and 'เพื่อทราบ'")
        if not success_criteria["quality_score"]:
            print("   - Improve overall content quality and relevance")
        if not success_criteria["type_accuracy"]:
            print("   - Enhance document type classification")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    success = run_context_engineered_test()
    sys.exit(0 if success else 1)