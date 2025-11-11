#!/usr/bin/env python3
"""
Debug script to test document similarity with the travel approval document
"""

import sys
import os
sys.path.append('src')

def test_document_similarity():
    """Test similarity with travel approval document"""
    print("🔍 Debugging Document Similarity")
    print("=" * 50)

    # Test query from the logs
    test_query = "เขียนบันทึกข้อความ ขออนุมัติเดินทางไปตรวจสอบตามแผนประจำปี 2568 จังหวัดบุรีรัมย์ 1-2 มกราคม 2568"

    try:
        from langchain.schema import Document
        from langchain_cohere import CohereEmbeddings
        from dotenv import load_dotenv

        load_dotenv()

        # Load the travel approval document
        with open('data_markdown/รายงานอนุมัติเดินทาง.md', 'r', encoding='utf-8') as f:
            travel_doc_content = f.read()

        travel_doc = Document(
            page_content=travel_doc_content,
            metadata={'source': 'data_markdown/รายงานอนุมัติเดินทาง.md'}
        )

        print(f"✅ Loaded travel document: {len(travel_doc_content)} chars")

        # Create embedding model
        embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0")
        print("✅ Embedding model created")

        # Compute embeddings
        doc_embedding = embedding_model.embed_documents([travel_doc_content])[0]
        query_embedding = embedding_model.embed_query(test_query)
        print("✅ Embeddings computed")

        # Compute similarity using cosine similarity (without scikit-learn)
        import numpy as np

        def cosine_similarity_manual(a, b):
            """Manual cosine similarity calculation"""
            a = np.array(a)
            b = np.array(b)
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return dot_product / (norm_a * norm_b)

        similarity = cosine_similarity_manual(doc_embedding, query_embedding)

        print(f"📊 Similarity Score: {similarity:.4f}")
        print(f"🎯 Threshold: 0.7500")
        print(f"✅ Would be similar: {similarity >= 0.75}")

        # Test different thresholds
        thresholds = [0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
        print(f"\n🔍 Testing different thresholds:")
        for threshold in thresholds:
            would_match = similarity >= threshold
            print(f"   {threshold}: {'✅ MATCH' if would_match else '❌ No match'}")

        # Test keywords matching
        doc_lower = travel_doc_content.lower()
        query_lower = test_query.lower()

        keywords = ["อนุมัติเดินทาง", "บุรีรัมย์", "ตรวจสอบ", "แผนประจำปี", "บันทึกข้อความ"]
        print(f"\n🔍 Keyword matching:")
        for keyword in keywords:
            in_doc = keyword in doc_lower
            in_query = keyword in query_lower
            print(f"   '{keyword}': doc={in_doc}, query={in_query}, match={in_doc and in_query}")

        return similarity

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_document_type_classification():
    """Test document type classification"""
    print("\n📋 Testing Document Type Classification")
    print("=" * 50)

    try:
        from src.processors.thai_document_templates import ThaiDocumentStructureAnalyzer

        analyzer = ThaiDocumentStructureAnalyzer()

        # Test query
        test_query = "เขียนบันทึกข้อความ ขออนุมัติเดินทางไปตรวจสอบตามแผนประจำปี 2568 จังหวัดบุรีรัมย์ 1-2 มกราคม 2568"

        result = analyzer.classify_document_type(test_query)
        print(f"🎯 Predicted type: {result['predicted_type']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")

        # Test with travel document content
        with open('data_markdown/รายงานอนุมัติเดินทาง.md', 'r', encoding='utf-8') as f:
            doc_content = f.read()

        doc_result = analyzer.classify_document_type(doc_content)
        print(f"📄 Document type: {doc_result['predicted_type']}")
        print(f"📊 Document confidence: {doc_result['confidence']:.2f}")

        # Check if they match
        types_match = result['predicted_type'] == doc_result['predicted_type']
        print(f"✅ Types match: {types_match}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    similarity = test_document_similarity()
    test_document_type_classification()

    print(f"\n🎯 SUMMARY:")
    if similarity is not None:
        if similarity >= 0.75:
            print("✅ Document should have been found!")
        else:
            print(f"⚠️  Similarity score ({similarity:.3f}) below threshold (0.75)")
            print("💡 Consider lowering the threshold or improving keyword matching")