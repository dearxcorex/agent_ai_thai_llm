#!/usr/bin/env python3
"""
Test the improved similarity engine with lowered threshold
"""

import sys
sys.path.append('src')

def test_improved_similarity():
    print("🚀 Testing Improved Similarity Engine")
    print("=" * 50)

    try:
        from langchain.schema import Document
        from src.processors.document_similarity import DocumentSimilarityEngine
        from dotenv import load_dotenv

        load_dotenv()

        # Test query (same as from the logs)
        test_query = "เขียนบันทึกข้อความ ขออนุมัติเดินทางไปตรวจสอบตามแผนประจำปี 2568 จังหวัดบุรีรัมย์ 1-2 มกราคม 2568"

        # Load all documents from data_markdown
        from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

        loader = DirectoryLoader(
            "data_markdown",
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            recursive=True
        )

        documents = loader.load()
        print(f"📄 Loaded {len(documents)} documents")

        # Create similarity engine
        engine = DocumentSimilarityEngine(documents)
        print("✅ Similarity engine created")

        # Test with lowered threshold
        threshold = 0.60
        is_similar, similar_docs = engine.check_similarity(test_query, threshold)

        print(f"\n🎯 Results:")
        print(f"   Similar documents found: {is_similar}")
        print(f"   Number of similar docs: {len(similar_docs)}")
        print(f"   Threshold used: {threshold}")

        if similar_docs:
            print(f"\n📋 Top similar documents:")
            for i, doc_info in enumerate(similar_docs[:3]):
                print(f"   {i+1}. {doc_info['metadata']['filename']}")
                print(f"      Original similarity: {doc_info.get('similarity', 'N/A'):.3f}")
                print(f"      Boosted similarity: {doc_info.get('boosted_similarity', 'N/A'):.3f}")
                print(f"      Keyword boost: {doc_info.get('keyword_boost', 'N/A'):.3f}")

                # Check if it's the travel document
                if 'อนุมัติเดินทาง' in doc_info['metadata']['filename']:
                    print(f"      ✅ FOUND THE TRAVEL DOCUMENT!")
                    return True

        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    found = test_improved_similarity()
    print(f"\n🎯 RESULT: {'✅ Travel document found!' if found else '❌ Travel document not found'}")