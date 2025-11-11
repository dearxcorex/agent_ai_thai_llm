#!/usr/bin/env python3
"""
Quick test of the full system to confirm similarity check works
"""

import sys
sys.path.append('src')

def test_full_system():
    print("🧪 Testing Full RAG System")
    print("=" * 40)

    try:
        from src.agents.context_engineered_rag_simple import app

        # Test query from the original logs
        test_query = "เขียนบันทึกข้อความ ขออนุมัติเดินทางไปตรวจสอบตามแผนประจำปี 2568 จังหวัดบุรีรัมย์ 1-2 มกราคม 2568"

        input_state = {
            "question": test_query,
            "web_search_needed": False,
            "query_rewrite_needed": False,
            "regeneration_needed": False,
            "warnings": [],
            # Initialize similarity fields
            "similar_documents_found": False,
            "similar_documents_by_type": [],
            "use_local_documents": False,
            "similarity_threshold": 0.60
        }

        run_config = {
            "configurable": {"thread_id": "test_session"},
            "recursion_limit": 50
        }

        print(f"🔍 Query: {test_query[:50]}...")
        print(f"⚙️  Similarity threshold: {input_state['similarity_threshold']}")

        # Run just the first few nodes to test similarity
        similarity_found = False
        web_search_triggered = False

        for output in app.stream(input_state, run_config):
            for node_name, node_output in output.items():
                print(f"\n📦 Node: '{node_name}'")

                if node_name == "check_similarity":
                    similar_found = node_output.get('similar_documents_found', False)
                    similar_count = len(node_output.get('similar_documents_by_type', []))
                    use_local = node_output.get('use_local_documents', False)
                    web_search = node_output.get('web_search_needed', False)

                    print(f"   Similar Documents Found: {'✅' if similar_found else '❌'}")
                    print(f"   Similar by Type: {similar_count}")
                    print(f"   Using Local Docs: {'✅' if use_local else '❌'}")
                    print(f"   Web Search Needed: {'✅' if web_search else '❌'}")

                    similarity_found = similar_found
                    if web_search:
                        web_search_triggered = True

                elif node_name == "web_search":
                    web_search_triggered = True
                    print("   ❌ WEB SEARCH EXECUTED!")

                elif node_name == "retrieve":
                    method = node_output.get('retrieval_method', 'unknown')
                    print(f"   Retrieval method: {method}")
                    if method == "similarity_match":
                        print("   ✅ USING SIMILAR DOCUMENTS!")

                # Stop after similarity check or web search to save time
                if node_name in ["web_search", "retrieve"]:
                    break

        print(f"\n🎯 FINAL RESULT:")
        print(f"   Similarity check worked: {'✅' if similarity_found else '❌'}")
        print(f"   Web search avoided: {'✅' if not web_search_triggered else '❌'}")

        return similarity_found and not web_search_triggered

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_system()
    print(f"\n🏁 SYSTEM TEST: {'✅ SUCCESS - Uses local docs!' if success else '❌ FAILED - Still using web search'}")