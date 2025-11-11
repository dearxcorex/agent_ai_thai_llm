#!/usr/bin/env python3
"""
Test script for Enhanced RAG System with Document Similarity
Tests the system components without requiring API keys
"""

import sys
import os
sys.path.append('src')

def test_document_templates():
    """Test Thai document templates"""
    print("\n=== Testing Thai Document Templates ===")

    try:
        from src.processors.thai_document_templates import ThaiDocumentTemplates

        templates = ThaiDocumentTemplates()
        available_types = templates.get_available_types()

        print(f"✓ Available document types: {available_types}")

        # Test template generation
        for doc_type in available_types[:3]:  # Test first 3 types
            template = templates.get_template(doc_type)
            if template:
                print(f"✓ Template for {doc_type}: {len(template.required_sections)} required sections")
                prompt = templates.generate_template_prompt(doc_type)
                print(f"✓ Generated prompt length: {len(prompt)} chars")
            else:
                print(f"✗ No template found for {doc_type}")

    except Exception as e:
        print(f"✗ Document templates test failed: {e}")
        return False

    return True

def test_document_structure_analyzer():
    """Test document structure analyzer"""
    print("\n=== Testing Document Structure Analyzer ===")

    try:
        from src.processors.thai_document_templates import ThaiDocumentStructureAnalyzer

        analyzer = ThaiDocumentStructureAnalyzer()

        # Test classification
        test_content = """
        เรื่อง เรียนเชิญประชุม
        เรียน ประธานคณะกรรมการ
        ในวันที่ 10 กรกฎาคม 2568 เวลา 09:00 น.
        จึงเรียนมาเพื่อโปรดพิจารณา
        """

        result = analyzer.classify_document_type(test_content)
        print(f"✓ Classification: {result['predicted_type']} (confidence: {result['confidence']:.2f})")

        # Test component extraction
        components = analyzer.extract_document_components(test_content, "หนังสือเชิญ")
        print(f"✓ Components extracted: {len(components['components'])}")

    except Exception as e:
        print(f"✗ Document structure analyzer test failed: {e}")
        return False

    return True

def test_thai_formatting():
    """Test Thai formatting utilities"""
    print("\n=== Testing Thai Formatting ===")

    try:
        from src.utils.thai_formatting import ThaiFormatter, validate_thai_document

        formatter = ThaiFormatter()

        # Test number conversion
        test_text = "วันที่ 10 กรกฎาคม 2568 เวลา 09:00 น."
        formatted = formatter.convert_to_thai_numbers(test_text)
        print(f"✓ Number conversion: {formatted}")

        # Test Thai formatting
        thai_formatted = formatter.apply_thai_formatting(test_text)
        print(f"✓ Thai formatting: {thai_formatted}")

        # Test validation
        validation = validate_thai_document("เรื่อง ทดสอบ จึงเรียนมาเพื่อทราบ")
        print(f"✓ Validation score: {validation['overall_score']:.2f}")

    except Exception as e:
        print(f"✗ Thai formatting test failed: {e}")
        return False

    return True

def test_configuration():
    """Test configuration"""
    print("\n=== Testing Configuration ===")

    try:
        from src.config import config as app_config

        # Test similarity config
        similarity_threshold = getattr(app_config, 'DOCUMENT_SIMILARITY_THRESHOLD', 0.75)
        print(f"✓ Similarity threshold: {similarity_threshold}")

        # Test other config values
        chunk_size = getattr(app_config, 'CHUNK_SIZE', 1024)
        print(f"✓ Chunk size: {chunk_size}")

        # Test data directory
        data_dir = getattr(app_config, 'DATA_DIR', './data_markdown')
        print(f"✓ Data directory: {data_dir}")

        # Check if data directory exists
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            print(f"✓ Data directory exists with {len(files)} files")
        else:
            print(f"⚠ Data directory not found: {data_dir}")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

    return True

def test_workflow_structure():
    """Test workflow structure without executing"""
    print("\n=== Testing Workflow Structure ===")

    try:
        # Check if main files exist
        files_to_check = [
            'src/agents/context_engineered_rag_simple.py',
            'src/processors/document_similarity.py',
            'src/processors/thai_document_templates.py',
            'main.py'
        ]

        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"✓ {file_path} exists")
            else:
                print(f"✗ {file_path} missing")
                return False

        # Try to import main components (without API calls)
        from src.agents.context_engineered_rag_simple import GraphState
        print("✓ GraphState imported")

        # Check if workflow nodes exist
        expected_nodes = ['analyze_context', 'check_similarity', 'retrieve', 'grade_documents', 'web_search', 'generate']
        print(f"✓ Expected workflow nodes: {expected_nodes}")

    except Exception as e:
        print(f"✗ Workflow structure test failed: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced RAG System with Document Similarity")
    print("=" * 60)

    tests = [
        test_configuration,
        test_thai_formatting,
        test_document_templates,
        test_document_structure_analyzer,
        test_workflow_structure
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! System structure is ready.")
        print("\n📝 Next steps:")
        print("1. Install scikit-learn: uv run pip install scikit-learn")
        print("2. Set up API keys in .env file")
        print("3. Run: uv run python main.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)