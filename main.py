#!/usr/bin/env python3
"""
Context Engineered RAG System for Thai Government Document Generation
Main entry point for the system
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_search_mode_choice():
    """Get user's search mode choice"""
    from src.config import config as app_config

    print("\n🔍 Search Mode Selection:")
    print("=" * 50)

    modes = app_config.SEARCH_MODES
    mode_list = list(modes.keys())

    for i, (mode, description) in enumerate(modes.items(), 1):
        default_marker = " (DEFAULT)" if mode == app_config.DEFAULT_SEARCH_MODE else ""
        print(f"  {i}. {mode.upper()}{default_marker}")
        print(f"     {description}")
        print()

    while True:
        try:
            choice = input(f"Select search mode (1-{len(mode_list)}) [default: {app_config.DEFAULT_SEARCH_MODE}]: ").strip()

            # Use default if user presses Enter
            if not choice:
                return app_config.DEFAULT_SEARCH_MODE

            # Validate choice
            if choice.isdigit() and 1 <= int(choice) <= len(mode_list):
                return mode_list[int(choice) - 1]
            else:
                # Allow direct mode name input
                if choice.lower() in mode_list:
                    return choice.lower()

            print(f"Invalid choice. Please enter 1-{len(mode_list)} or mode name.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def main():
    """Main execution function"""
    print("🚀 Enhanced RAG System with Web Search Options")
    print("=" * 50)

    # Get search mode choice first (before any expensive operations)
    search_mode = get_search_mode_choice()
    if search_mode is None:
        return

    print(f"\n✅ Selected search mode: {search_mode.upper()}")

    # Now initialize the system (only after user has made choices)
    logger.info("Initializing RAG system...")

    # Import the main agent (delayed until after user input)
    from agents.context_engineered_rag_simple import app
    logger.info("✅ RAG system initialized")

    # Test queries
    test_cases = [
        {
            "name": "Meeting Invitation - ITU Call Center Prevention",
            "query": "สร้างหนังสือเชิญประชุมกับคณะกรรมการ ITU เรื่องการป้องกันแก๊ง call center ในประเทศเพื่อนบ้าน โดยจะเสนอมาตรฐานการลดกำลังส่ง cellsite กำหนดวันที่ 10 กรกฎาคม 2568 เวลา 09:00 น. ณ ห้องประชุมใหญ่ สำนักงาน กสทช",
            "recommended_modes": ["local", "hybrid"]
        },
        {
            "name": "Travel Approval Memo",
            "query": "เขียนบันทึกข้อความ ขออนุมัติเดินทางไปตรวจสอบตามแผนประจำปี 2568 จังหวัดบุรีรัมย์ 1-2 มกราคม 2568",
            "recommended_modes": ["local", "hybrid", "compare"]
        },
        {
            "name": "Inspection Report",
            "query": "จัดทำรายงานการตรวจสอบสถานีวิทยุคมนาคมประจำปี 2568 ตามแผนปฏิบัติการที่ได้รับมอบหมาย",
            "recommended_modes": ["local", "hybrid"]
        },
        {
            "name": "Cell Radius Report",
            "query": "จัดทำรายงานเรื่อง cell radius อำเภอ สายตะกู จังหวัดบุรีรัมย์ ว่าสัญญาญไม่สามารถข้ามไปประเทศเพื่อนบ้านได้",
            "recommended_modes": ["web", "hybrid", "compare"]
        }
    ]

    # Let user choose test case
    print("\n📋 Available Test Cases:")
    for i, case in enumerate(test_cases, 1):
        recommended = case.get("recommended_modes", [])
        mode_status = "✅" if search_mode in recommended else "⚠️"
        print(f"  {i}. {case['name']} {mode_status}")
        if recommended:
            print(f"     Recommended modes: {', '.join(recommended)}")

    while True:
        try:
            choice = input(f"\nSelect test case (1-{len(test_cases)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(test_cases):
                test_case = test_cases[int(choice) - 1]
                break
            print("Invalid choice. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return

    logger.info(f"Running test case: {test_case['name']}")
    logger.info(f"Query: {test_case['query'][:50]}...")

    # Prepare input state
    input_state = {
        "question": test_case["query"],
        "search_mode": search_mode,  # Add search mode to state
        "web_search_needed": False,
        "query_rewrite_needed": False,
        "regeneration_needed": False,
        "warnings": [],
        # Initialize similarity fields
        "similar_documents_found": False,
        "similar_documents_by_type": [],
        "use_local_documents": False,
        "similarity_threshold": 0.75,
        # Results storage for comparison mode
        "local_generation": None,
        "web_generation": None,
        "final_generation": None
    }

    # Configure execution
    run_config = {
        "configurable": {"thread_id": "main_session"},
        "recursion_limit": 50
    }

    # Execute workflow
    final_state = None
    last_generation = None
    local_generation = None
    web_generation = None

    try:
        for output in app.stream(input_state, run_config):
            for node_name, node_output in output.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Node: '{node_name}' completed")
                if node_name == "analyze_context":
                    logger.info(f"  Document Type: {node_output.get('document_type')}")
                    logger.info(f"  Extracted Info: {len(node_output.get('extracted_info', {}))} categories")
                elif node_name == "check_similarity":
                    similar_found = node_output.get('similar_documents_found', False)
                    similar_count = len(node_output.get('similar_documents_by_type', []))
                    use_local = node_output.get('use_local_documents', False)
                    logger.info(f"  Similar Documents Found: {similar_found}")
                    logger.info(f"  Similar by Type: {similar_count}")
                    logger.info(f"  Using Local Docs: {use_local}")
                elif node_name == "generate":
                    gen_length = len(node_output.get('generation', ''))
                    local_gen = node_output.get('local_generation')
                    web_gen = node_output.get('web_generation')

                    logger.info(f"  Generated Document ({gen_length} chars)")
                    if local_gen:
                        logger.info(f"  Local Generation: {len(local_gen)} chars")
                        local_generation = local_gen
                    if web_gen:
                        logger.info(f"  Web Generation: {len(web_gen)} chars")
                        web_generation = web_gen

                logger.info(f"{'='*60}")

                if 'generation' in node_output:
                    last_generation = node_output['generation']
                final_state = node_output

        # Update final state with captured generations for COMPARE mode
        if search_mode == "compare":
            if final_state:
                final_state['local_generation'] = local_generation
                final_state['web_generation'] = web_generation

        # Display results
        display_results(test_case, final_state, last_generation, search_mode)

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1

    return 0

def display_results(test_case, final_state, generation, search_mode="local"):
    """Display formatted results"""

    # For COMPARE mode, show both local and web results
    if search_mode == "compare" and final_state:
        local_gen = final_state.get('local_generation')
        web_gen = final_state.get('web_generation')

        print("\n" + "="*80)
        print("COMPARISON MODE RESULTS")
        print("="*80)

        # Local Results
        print("\n" + "📚 LOCAL DOCUMENTS RESULT")
        print("-" * 40)
        if local_gen:
            print(local_gen)
        else:
            print("(No local generation found)")

        # Web Results
        print("\n" + "🌐 WEB SEARCH RESULT")
        print("-" * 40)
        if web_gen:
            print(web_gen)
        else:
            print("(No web generation found)")

        # Comparison note
        print("\n" + "="*80)
        print("💡 COMPARISON NOTE")
        print("="*80)
        print("• Local result uses similar documents from data_markdown/")
        print("• Web result uses latest information from internet")
        print("• Choose the result that best fits your requirements")
        print("="*80)

        return

    # For other modes, show single result
    print("\n" + "="*80)
    print("FINAL GENERATED DOCUMENT")
    print("="*80)

    if generation:
        print(generation)
    elif final_state and 'generation' in final_state:
        print(final_state['generation'])
    else:
        print("(No generation found)")

    # Detailed results
    if final_state:
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)

        # Context Engineering Results
        if 'document_type' in final_state:
            print(f"\n📋 Document Type: {final_state['document_type']}")
        if 'extracted_info' in final_state:
            info = final_state['extracted_info']
            print(f"📊 Extracted Information:")
            for key, value in info.items():
                if value:
                    print(f"   {key}: {value}")

        # Document Similarity Results
        if 'similar_documents_found' in final_state:
            similar_found = final_state['similar_documents_found']
            use_local = final_state.get('use_local_documents', False)
            similar_count = len(final_state.get('similar_documents_by_type', []))
            print(f"\n🔍 Document Similarity Results:")
            print(f"   Similar Documents Found: {'✅' if similar_found else '❌'}")
            print(f"   Documents by Type: {similar_count}")
            print(f"   Using Local Documents: {'✅' if use_local else '❌'}")
            if similar_found and final_state.get('similar_documents_by_type'):
                print(f"   Similar Files:")
                for doc in final_state['similar_documents_by_type'][:3]:
                    filename = doc.metadata.get('source', '').split('/')[-1]
                    print(f"     • {filename}")

        # Retrieval Results
        if 'retrieval_method' in final_state:
            print(f"\n🔄 Retrieval Method: {final_state['retrieval_method']}")

        # Thai Format Validation
        if generation:
            from utils.thai_formatting import validate_thai_document
            thai_validation = validate_thai_document(generation)
            print(f"\n🇹🇭 Thai Format Validation:")
            print(f"   Compliance Score: {thai_validation['overall_score']:.2f}")
            print(f"   Is Compliant: {'✅' if thai_validation['is_compliant'] else '❌'}")
            if thai_validation['issues']:
                print(f"   Issues: {', '.join(thai_validation['issues'])}")

        # Warnings
        if 'warnings' in final_state and final_state['warnings']:
            print(f"\n⚠️ Warnings:")
            for warning in final_state['warnings']:
                print(f"   • {warning}")

    print("\n" + "="*80)
    print("✅ Generation Complete")
    print("="*80)

if __name__ == "__main__":
    sys.exit(main())