#!/usr/bin/env python3
"""
Document Similarity Engine for Thai RAG System
Checks for similar documents in local corpus before retrieval
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from langchain.schema import Document
from langchain_cohere import CohereEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DocumentSimilarityEngine:
    """
    Engine for checking document similarity before retrieval
    """

    def __init__(self, documents: List[Document], embedding_model: str = "embed-multilingual-v3.0"):
        """
        Initialize similarity engine

        Args:
            documents: List of documents to check against
            embedding_model: Cohere embedding model name
        """
        self.documents = documents
        self.embedding_model = CohereEmbeddings(model=embedding_model)
        self.document_embeddings = None
        self.doc_metadata = []
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Pre-compute embeddings for all documents"""
        logger.info("Computing document embeddings for similarity checking...")

        # Extract metadata
        self.doc_metadata = []
        for doc in self.documents:
            metadata = {
                "source": doc.metadata.get("source", ""),
                "filename": Path(doc.metadata.get("source", "")).name,
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            self.doc_metadata.append(metadata)

        # Compute embeddings for full documents
        doc_texts = [doc.page_content for doc in self.documents]
        try:
            self.document_embeddings = self.embedding_model.embed_documents(doc_texts)
            logger.info(f"✓ Computed embeddings for {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            self.document_embeddings = None

    def check_similarity(self, query: str, threshold: float = 0.75) -> Tuple[bool, List[Dict]]:
        """
        Check if query is similar to any documents

        Args:
            query: User query
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            Tuple of (is_similar, similar_documents_info)
        """
        if self.document_embeddings is None:
            logger.warning("No embeddings available, cannot check similarity")
            return False, []

        try:
            # Embed the query
            query_embedding = self.embedding_model.embed_query(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)

            # Compute similarities
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]

            # Find documents above threshold
            similar_docs = []
            max_similarity = 0.0

            # Extract keywords from query for boosting
            query_keywords = set(query.lower().split())

            for i, similarity in enumerate(similarities):
                # Apply keyword matching boost for Thai documents
                doc_content = self.documents[i].page_content.lower()
                doc_keywords = set(doc_content.split())

                # Calculate keyword overlap
                common_keywords = query_keywords.intersection(doc_keywords)
                keyword_boost = len(common_keywords) / max(len(query_keywords), 1)

                # Apply boost (up to 0.1 increase)
                boosted_similarity = min(similarity + (keyword_boost * 0.1), 1.0)

                if boosted_similarity >= threshold:
                    doc_info = {
                        "index": i,
                        "similarity": float(similarity),
                        "boosted_similarity": float(boosted_similarity),
                        "keyword_boost": float(keyword_boost),
                        "metadata": self.doc_metadata[i],
                        "document": self.documents[i]
                    }
                    similar_docs.append(doc_info)

                max_similarity = max(max_similarity, boosted_similarity)

            # Sort by boosted similarity (descending)
            similar_docs.sort(key=lambda x: x["boosted_similarity"], reverse=True)

            is_similar = len(similar_docs) > 0

            logger.info(f"Similarity check: max={max_similarity:.3f}, threshold={threshold}, found={len(similar_docs)} similar docs")

            return is_similar, similar_docs

        except Exception as e:
            logger.error(f"Similarity check failed: {e}")
            return False, []

    def get_similar_documents_by_type(self, query: str, document_type: str, threshold: float = 0.6) -> List[Document]:
        """
        Get similar documents of a specific type

        Args:
            query: User query
            document_type: Target document type
            threshold: Similarity threshold (lowered to match main threshold)

        Returns:
            List of similar documents of specified type
        """
        is_similar, similar_docs = self.check_similarity(query, threshold)

        if not is_similar:
            return []

        # Filter by document type (based on filename patterns)
        type_keywords = {
            "หนังสือเชิญ": ["เชิญ", "เชิญประชุม", "เชิญตรวจ"],
            "รายงาน": ["รายงาน", "report"],
            "บันทึกข้อความ": ["บันทึก", "ข้อความ", "อนุมัติ", "เดินทาง"],  # Added more keywords
            "หนังสือร้องขอ": ["ร้องขอ", "ขอ"],
            "หนังสืออนุมัติ": ["อนุมัติ", "approve"]
        }

        # Get keywords for the target type
        keywords = type_keywords.get(document_type, [])
        if not keywords:
            # If no keywords found, return all similar docs
            return [doc_info["document"] for doc_info in similar_docs]

        # Filter by keywords in filename or content
        filtered_docs = []
        for doc_info in similar_docs:
            filename = doc_info["metadata"]["filename"].lower()
            content = doc_info["document"].page_content.lower()

            if any(keyword in filename or keyword in content for keyword in keywords):
                filtered_docs.append(doc_info["document"])

        return filtered_docs

    def analyze_document_types(self) -> Dict[str, int]:
        """
        Analyze document types in the corpus

        Returns:
            Dictionary of document type counts
        """
        type_patterns = {
            "หนังสือเชิญ": ["เชิญ"],
            "รายงาน": ["รายงาน"],
            "บันทึกข้อความ": ["บันทึก"],
            "หนังสือร้องขอ": ["ร้องขอ", "ขอ"],
            "หนังสืออนุมัติ": ["อนุมัติ"],
            "แจ้งแผน": ["แจ้ง", "แผน"],
            "คลื่นความถี่": ["คลื่น", "ความถี่"],
            "อื่นๆ": []
        }

        type_counts = {doc_type: 0 for doc_type in type_patterns.keys()}

        for doc in self.documents:
            content_lower = doc.page_content.lower()
            filename_lower = Path(doc.metadata.get("source", "")).name.lower()

            classified = False
            for doc_type, patterns in type_patterns.items():
                if doc_type == "อื่นๆ":
                    continue
                if any(pattern in content_lower or pattern in filename_lower for pattern in patterns):
                    type_counts[doc_type] += 1
                    classified = True
                    break

            if not classified:
                type_counts["อื่นๆ"] += 1

        return type_counts

    def get_similarity_statistics(self) -> Dict:
        """
        Get statistics about document similarities

        Returns:
            Dictionary with similarity statistics
        """
        if self.document_embeddings is None:
            return {"error": "No embeddings available"}

        try:
            # Compute pairwise similarities
            similarities = cosine_similarity(self.document_embeddings)

            # Get upper triangle (excluding diagonal)
            upper_tri = similarities[np.triu_indices_from(similarities, k=1)]

            stats = {
                "total_documents": len(self.documents),
                "mean_similarity": float(np.mean(upper_tri)),
                "max_similarity": float(np.max(upper_tri)),
                "min_similarity": float(np.min(upper_tri)),
                "std_similarity": float(np.std(upper_tri))
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to compute similarity statistics: {e}")
            return {"error": str(e)}

def create_similarity_engine(documents: List[Document]) -> DocumentSimilarityEngine:
    """
    Factory function to create similarity engine

    Args:
        documents: List of documents

    Returns:
        DocumentSimilarityEngine instance
    """
    return DocumentSimilarityEngine(documents)

# Test function
if __name__ == "__main__":
    # Simple test
    from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

    logger.basicConfig(level=logging.INFO)

    # Load documents
    loader = DirectoryLoader(
        "/Users/deardevx/Documents/my_stufF/master_project/research_llm/agent/data_markdown",
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        recursive=True
    )

    docs = loader.load()

    # Create similarity engine
    engine = DocumentSimilarityEngine(docs)

    # Test similarity
    test_queries = [
        "เขียนหนังสือเชิญประชุม",
        "ทำรายงานการตรวจสอบ",
        "ขออนุมัติเดินทาง"
    ]

    for query in test_queries:
        is_similar, similar_docs = engine.check_similarity(query)
        print(f"\nQuery: {query}")
        print(f"Similar: {is_similar}")
        if similar_docs:
            for doc_info in similar_docs[:3]:
                print(f"  - {doc_info['metadata']['filename']} (similarity: {doc_info['similarity']:.3f})")

    # Print document type analysis
    print(f"\nDocument Types: {engine.analyze_document_types()}")
    print(f"Similarity Statistics: {engine.get_similarity_statistics()}")