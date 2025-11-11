"""
Simplified Context Engineered RAG System
Fixed version without complex template variable issues
"""

from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

from langgraph.graph import END, StateGraph, START

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
from typing_extensions import TypedDict
import operator
from dotenv import load_dotenv
import logging
import json

# Import configuration
from src.config import config as app_config

# Import context engineering system
from src.processors.context_engineering_system import ThaiDocumentContextEngineer, create_context_engineer
from src.utils.thai_formatting import ThaiFormatter, validate_thai_document

# Import document similarity engine
from src.processors.document_similarity import DocumentSimilarityEngine, create_similarity_engine

# Import document templates
from src.processors.thai_document_templates import ThaiDocumentTemplates, create_thai_document_templates

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Validate configuration
app_config.validate_config()

# ==================== Enhanced State Schema ====================

class GraphState(TypedDict):
    """
    Enhanced state schema for context engineered RAG system with document similarity and web search options
    """
    # Original query
    question: str

    # Search Mode
    search_mode: str  # local, web, hybrid, compare

    # Context engineering
    document_type: Optional[str]
    extracted_info: Optional[Dict]

    # Document Similarity
    similar_documents_found: bool
    similar_documents_by_type: List[Document]
    use_local_documents: bool
    similarity_threshold: float

    # Retrieval
    documents: List[Document]
    filtered_docs: List[Document]
    reranked_docs: List[Document]

    # Generation Results
    generation: Optional[str]
    local_generation: Optional[str]  # For comparison mode
    web_generation: Optional[str]     # For comparison mode
    final_generation: Optional[str]   # Selected final result
    generation_phase: Optional[str]   # For COMPARE mode: "local" or "web" or None

    # Control flow
    web_search_needed: bool
    query_rewrite_needed: bool
    regeneration_needed: bool

    # Metadata
    error: Optional[str]
    warnings: Annotated[List[str], operator.add]

# ==================== Document Loading and Processing ====================

logger.info("Loading documents...")

# Load documents from markdown files
loader = DirectoryLoader(
    app_config.DATA_DIR,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    recursive=True
)

documents = loader.load()
logger.info(f"Loaded {len(documents)} documents")

# Chunk documents
chunker = RecursiveCharacterTextSplitter(
    chunk_size=app_config.CHUNK_SIZE,
    chunk_overlap=app_config.CHUNK_OVERLAP,
    length_function=len,
    separators=app_config.SEPARATORS
)

doc_splits = chunker.split_documents(documents)
logger.info(f"Split into {len(doc_splits)} chunks")

# ==================== Hybrid Retrieval Setup ====================

logger.info("Setting up hybrid retrieval system...")

# 1. Dense Retriever (FAISS + Cohere Embeddings)
dense_vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=CohereEmbeddings(model=app_config.EMBEDDING_MODEL)
)
dense_retriever = dense_vectorstore.as_retriever(
    search_kwargs={"k": app_config.DENSE_RETRIEVAL_K}
)

# 2. BM25 Retriever (Sparse)
bm25_retriever = BM25Retriever.from_documents(doc_splits)
bm25_retriever.k = app_config.BM25_RETRIEVAL_K

# 3. Ensemble Retriever (Hybrid with RRF)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# 4. Reranker
reranker = CohereRerank(
    model=app_config.RERANK_MODEL,
    top_n=app_config.RERANK_TOP_K
)

logger.info("✓ Hybrid retrieval system ready")

# ==================== LLM Setup ====================

# DeepSeek for generation
logger.info("Initializing DeepSeek LLM...")

llm_generation = ChatOpenAI(
    model=app_config.GENERATION_MODEL,
    base_url="https://api.deepseek.com",
    api_key=app_config.DEEPSEEK_API_KEY,
    temperature=app_config.GENERATION_TEMPERATURE
)

# Context engineering LLM (DeepSeek with JSON mode)
llm_structured = ChatOpenAI(
    model=app_config.GRADING_MODEL,
    base_url="https://api.deepseek.com",
    api_key=app_config.DEEPSEEK_API_KEY,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Initialize context engineering system
context_engineer = create_context_engineer()
thai_formatter = ThaiFormatter()

# Initialize document similarity engine
logger.info("Initializing document similarity engine...")
similarity_engine = DocumentSimilarityEngine(documents)

# Initialize document templates
logger.info("Initializing Thai document templates...")
thai_templates = create_thai_document_templates()

logger.info("✓ LLMs and context engineering system ready")

# ==================== Web Search Tool ====================

web_search_tool = TavilySearchResults(k=app_config.WEB_SEARCH_RESULTS)

# ==================== Agent 1: Context Engineering Analysis ====================

def analyze_query_context(state: GraphState) -> GraphState:
    """
    Analyze user query with context engineering
    """
    logger.info("--- CONTEXT ENGINEERING ANALYSIS ---")

    question = state["question"]

    try:
        # Classify document type
        document_type, confidence = context_engineer.classify_request(question)
        logger.info(f"Document classified as: {document_type} (confidence: {confidence:.2f})")

        # Extract structured information
        extracted_info = context_engineer.extract_document_info(question)
        logger.info(f"Extracted info: {len(extracted_info)} categories")

        # Generate optimized query
        optimized_query_parts = [question]
        if extracted_info.get("technical_terms"):
            optimized_query_parts.extend(extracted_info["technical_terms"])
        if extracted_info.get("action_words"):
            optimized_query_parts.extend(extracted_info["action_words"])

        optimized_query = " ".join(optimized_query_parts)

        logger.info(f"Optimized query: {optimized_query}")

        return {
            "document_type": document_type,
            "extracted_info": extracted_info,
            "optimized_query": optimized_query
        }

    except Exception as e:
        logger.error(f"Context analysis failed: {e}")
        return {
            "document_type": "บันทึกข้อความ",
            "extracted_info": {},
            "optimized_query": question,
            "warnings": [f"Context analysis failed: {str(e)}"]
        }


# ==================== Agent 1.5: Document Similarity Check ====================

def check_document_similarity(state: GraphState) -> GraphState:
    """
    Check if query matches documents in local corpus before retrieval
    """
    logger.info("--- DOCUMENT SIMILARITY CHECK ---")

    question = state.get("optimized_query") or state["question"]
    document_type = state.get("document_type", "บันทึกข้อความ")

    try:
        # Check similarity with threshold from config (use 0.75 as default)
        similarity_threshold = getattr(app_config, 'DOCUMENT_SIMILARITY_THRESHOLD', 0.75)
        is_similar, similar_docs = similarity_engine.check_similarity(question, similarity_threshold)

        # Get similar documents by type if similarity found
        similar_docs_by_type = []
        if is_similar:
            similar_docs_by_type = similarity_engine.get_similar_documents_by_type(
                question, document_type, threshold=similarity_threshold
            )

        logger.info(f"Similarity results: found={is_similar}, by_type={len(similar_docs_by_type)}")

        # Decision logic
        use_local_docs = is_similar and len(similar_docs_by_type) > 0
        web_search_needed = not use_local_docs

        return {
            "similar_documents_found": is_similar,
            "similar_documents_by_type": similar_docs_by_type,
            "use_local_documents": use_local_docs,
            "web_search_needed": web_search_needed,
            "similarity_threshold": similarity_threshold
        }

    except Exception as e:
        logger.error(f"Document similarity check failed: {e}")
        # Fallback to retrieval if similarity check fails
        return {
            "similar_documents_found": False,
            "similar_documents_by_type": [],
            "use_local_documents": False,
            "web_search_needed": True,
            "warnings": [f"Document similarity check failed: {str(e)}"]
        }


# ==================== Agent 2: Hybrid Retrieval ====================

def hybrid_retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents using hybrid approach or use similar documents if found
    """
    logger.info("--- HYBRID RETRIEVAL ---")

    # Use optimized query from context engineering
    query = state.get("optimized_query") or state["question"]
    document_type = state.get("document_type", "บันทึกข้อความ")
    use_local_docs = state.get("use_local_documents", False)
    similar_docs = state.get("similar_documents_by_type", [])

    try:
        # Check if we have similar documents from similarity check
        if use_local_docs and similar_docs:
            logger.info(f"Using {len(similar_docs)} similar documents found by similarity engine")

            # Use similar documents directly (no need to retrieve/rerank)
            return {
                "documents": similar_docs,
                "reranked_docs": similar_docs,
                "filtered_docs": similar_docs,  # Pre-filter for lenient grading
                "retrieval_method": "similarity_match"
            }

        # Standard hybrid retrieval when no similar documents found
        logger.info(f"No similar documents found, performing hybrid retrieval for {document_type}...")

        # Step 1: Ensemble retrieval (BM25 + Dense)
        logger.info("Retrieving with ensemble...")
        ensemble_docs = ensemble_retriever.invoke(query)
        logger.info(f"Retrieved {len(ensemble_docs)} documents from ensemble")

        # Step 2: Rerank with Cohere
        logger.info("Reranking documents...")
        docs_for_rerank = [doc.page_content for doc in ensemble_docs]
        reranked_results = reranker.rerank(query=query, documents=docs_for_rerank)

        # Handle different response formats
        if hasattr(reranked_results, 'results'):
            reranked_docs = [ensemble_docs[result['index']] for result in reranked_results.results]
        elif isinstance(reranked_results, list):
            reranked_docs = [ensemble_docs[result['index']] for result in reranked_results]
        else:
            reranked_docs = [ensemble_docs[result.index if hasattr(result, 'index') else result['index']] for result in reranked_results]

        logger.info(f"Reranked to top {len(reranked_docs)} documents")

        return {
            "documents": ensemble_docs,
            "reranked_docs": reranked_docs,
            "retrieval_method": "context_aware_hybrid"
        }

    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        return {
            "documents": [],
            "reranked_docs": [],
            "retrieval_method": "failed",
            "error": f"Retrieval failed: {str(e)}"
        }


# ==================== Agent 3: Document Grading ====================

def grade_documents(state: GraphState) -> GraphState:
    """
    Grade document relevance
    """
    logger.info("--- DOCUMENT GRADING ---")

    question = state.get("optimized_query") or state["question"]
    documents = state.get("reranked_docs", [])
    document_type = state.get("document_type", "บันทึกข้อความ")
    use_local_docs = state.get("use_local_documents", False)
    retrieval_method = state.get("retrieval_method", "")

    logger.info(f"Grading debug: use_local_docs={use_local_docs}, method={retrieval_method}")

    if not documents:
        logger.warning("No documents to grade")
        return {
            "filtered_docs": [],
            "web_search_needed": True,
            "query_rewrite_needed": False
        }

    # If documents came from similarity matching, be more lenient
    if use_local_docs:
        logger.info("Using lenient grading for similarity-matched documents")
        # Pass all similar documents through without filtering
        return {
            "filtered_docs": documents,
            "web_search_needed": False,
            "query_rewrite_needed": False
        }

    # Standard grading for other cases
    doc_grades = []
    filtered_docs = []

    for i, doc in enumerate(documents[:app_config.FINAL_CONTEXT_K]):
        try:
            # Simple confidence calculation based on content matching
            content_lower = doc.page_content.lower()
            query_lower = question.lower()

            # Count keyword matches
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            matches = len(query_words.intersection(content_words))
            confidence = matches / len(query_words) if query_words else 0.5

            doc_grades.append({
                "doc_index": i,
                "confidence": confidence,
                "reasoning": f"Found {matches} matching keywords"
            })

            logger.info(f"Doc {i+1}: Confidence={confidence:.2f}")

            # Filter documents above threshold
            if confidence >= app_config.LOW_CONFIDENCE_THRESHOLD:
                filtered_docs.append(doc)

        except Exception as e:
            logger.error(f"Failed to grade document {i}: {e}")
            doc_grades.append({
                "doc_index": i,
                "confidence": 0.5,
                "reasoning": f"Grading failed: {str(e)}"
            })

    # Calculate average confidence
    average_confidence = sum(g["confidence"] for g in doc_grades) / len(doc_grades) if doc_grades else 0.0

    logger.info(f"Average confidence: {average_confidence:.2f}")
    logger.info(f"Filtered docs: {len(filtered_docs)} out of {len(documents)}")

    # Routing logic
    web_search_needed = False
    query_rewrite_needed = False

    if average_confidence < app_config.VERY_LOW_THRESHOLD:
        logger.info("→ Very low confidence: triggering WEB SEARCH")
        web_search_needed = True
    elif average_confidence < app_config.LOW_CONFIDENCE_THRESHOLD:
        logger.info("→ Low confidence: triggering QUERY REWRITE")
        query_rewrite_needed = True
    elif len(filtered_docs) < app_config.MIN_RELEVANT_DOCS:
        logger.info("→ Insufficient relevant docs: triggering QUERY REWRITE")
        query_rewrite_needed = True
    else:
        logger.info("→ Sufficient confidence: proceeding to GENERATION")

    # Set generation phase for COMPARE mode
    generation_phase = None
    search_mode = state.get("search_mode", "local")
    if search_mode == "compare":
        use_local_docs = state.get("use_local_documents", False)
        local_generation = state.get("local_generation")
        web_generation = state.get("web_generation")

        if use_local_docs and not local_generation:
            generation_phase = "local"
        elif not use_local_docs and not web_generation:
            generation_phase = "web"

    result = {
        "filtered_docs": filtered_docs,
        "web_search_needed": web_search_needed,
        "query_rewrite_needed": query_rewrite_needed
    }

    if generation_phase:
        result["generation_phase"] = generation_phase

    return result


# ==================== Agent 4: Web Search ====================

def web_search(state: GraphState) -> GraphState:
    """
    Perform web search when document confidence is very low
    """
    logger.info("--- WEB SEARCH ---")

    question = state.get("optimized_query") or state["question"]
    document_type = state.get("document_type", "บันทึกข้อความ")

    try:
        # Prepare search query
        search_prompt = f"หนังสือราชการไทย {document_type} {question}"

        logger.info(f"Searching: {search_prompt}")
        search_results = web_search_tool.invoke(search_prompt)

        # Convert to documents
        web_docs = []
        for result in search_results:
            web_doc = Document(
                page_content=result["content"],
                metadata={"source": "web_search", "url": result.get("url", "")}
            )
            web_docs.append(web_doc)

        logger.info(f"Found {len(web_docs)} web results")

        # Combine with existing filtered docs
        existing_docs = state.get("filtered_docs", [])
        combined_docs = existing_docs + web_docs

        # Set generation phase for COMPARE mode when coming from web search
        result = {
            "filtered_docs": combined_docs,
            "web_search_needed": False
        }

        search_mode = state.get("search_mode", "local")
        if search_mode == "compare":
            result["generation_phase"] = "web"

        return result

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "web_search_needed": False,
            "warnings": [f"Web search failed: {str(e)}"]
        }


# ==================== Agent 5: Context-Aware Generation ====================

def generate_with_context(state: GraphState) -> GraphState:
    """
    Generate document with context engineering
    """
    logger.info("--- CONTEXT-ENGINEERED GENERATION ---")

    question = state["question"]
    documents = state.get("filtered_docs", [])
    document_type = state.get("document_type", "บันทึกข้อความ")
    extracted_info = state.get("extracted_info", {})

    if not documents:
        logger.warning("No documents available for generation")
        return {
            "generation": "ไม่สามารถสร้างเอกสารได้เนื่องจากไม่พบข้อมูลที่เกี่ยวข้อง กรุณาระบุรายละเอียดเพิ่มเติม",
            "regeneration_needed": True
        }

    # Create context
    context = "\n\n---\n\n".join([doc.page_content for doc in documents[:app_config.FINAL_CONTEXT_K]])

    # Get document template prompt
    template_prompt = thai_templates.generate_template_prompt(document_type)

    # Create enhanced generation prompt
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.GENERATION_SYSTEM_PROMPT + """

เป็นทางการ
เอกสารประเภท: """ + document_type + """

คำแนะนำเฉพาะ:
1. ใช้เลขไทย (๑, ๒, ๓) ในการลำดับข้อเท่านั้น
2. ใช้ปฏิทินพุทธศักราชเสมอ
3. แยกเนื้อหาส่วนที่ ๒ (เพื่อพิจารณา) และ ๓ (เพื่อทราบ) ให้ต่างกันชัดเจน
4. ใช้ภาษาราชการที่เป็นทางการ
5. รวมข้อมูลจากการสกัดจากคำขอต้นฉบับทั้งหมด
6. ปฏิบัติตามโครงสร้างและรูปแบบที่กำหนดในเทมเพลต

""" + template_prompt),
        ("human", """ข้อมูลอ้างอิง:
{context}

คำขอต้นฉบับ: {question}

ข้อมูลที่สกัดได้:
{extracted_info_json}

วิธีการดำเนินการ (ควรใช้ข้อมูลจาก similarity engine):
- ถ้าพบเอกสารที่คล้ายกัน: ใช้โครงสร้างและภาษาจากเอกสารนั้นเป็นหลัก
- ถ้าไม่พบเอกสารที่คล้ายกัน: ใช้เทมเพลตมาตรฐานตามประเภทเอกสาร

กรุณาสร้างเอกสารราชการประเภท {document_type} ตามโครงสร้างที่กำหนดและใช้ข้อมูลอ้างอิงที่มีอยู่ให้เป็นประโยชน์""")
    ])

    try:
        rag_chain = generation_prompt | llm_generation | StrOutputParser()
        generation = rag_chain.invoke({
            "context": context,
            "question": question,
            "document_type": document_type,
            "extracted_info_json": json.dumps(extracted_info, ensure_ascii=False, indent=2)
        })

        logger.info(f"Generated {document_type} ({len(generation)} chars)")

        # Store generation based on search mode and generation phase
        search_mode = state.get("search_mode", "local")
        generation_phase = state.get("generation_phase")

        if search_mode == "compare":
            # Use generation_phase to determine where to store the result
            if generation_phase == "local":
                return {
                    "local_generation": generation,
                    "generation": generation
                }
            elif generation_phase == "web":
                return {
                    "web_generation": generation,
                    "generation": generation
                }
            else:
                # Fallback: try to determine from context
                use_local_docs = state.get("use_local_documents", False)
                if use_local_docs:
                    return {
                        "local_generation": generation,
                        "generation": generation
                    }
                else:
                    return {
                        "web_generation": generation,
                        "generation": generation
                    }
        elif search_mode == "web":
            return {
                "web_generation": generation,
                "generation": generation
            }

        # Default for local and hybrid modes
        return {
            "generation": generation
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "generation": f"เกิดข้อผิดพลาดในการสร้างเอกสาร: {str(e)}",
            "error": str(e),
            "regeneration_needed": True
        }


# ==================== Enhanced Routing Functions ====================

def route_after_similarity_check(state: GraphState) -> str:
    """
    Route based on search mode and similarity check results
    """
    search_mode = state.get("search_mode", "local")
    web_search_needed = state.get("web_search_needed", False)
    use_local_docs = state.get("use_local_documents", False)

    logger.info(f"Routing decision - Mode: {search_mode}, Use Local: {use_local_docs}, Web Needed: {web_search_needed}")

    # Handle different search modes
    if search_mode == "web":
        # Web only mode - skip similarity and go directly to web search
        logger.info("→ WEB-ONLY MODE: Skipping similarity check, going to web search")
        return "web_search"
    elif search_mode == "compare":
        # Comparison mode - need both local and web results
        if use_local_docs:
            logger.info("→ COMPARE MODE: Getting local documents first")
            # Note: We'll set generation_phase in the grade_documents routing
            return "retrieve"
        else:
            logger.info("→ COMPARE MODE: Getting web documents first")
            # Note: We'll set generation_phase in the grade_documents routing
            return "web_search"
    elif web_search_needed and not use_local_docs:
        # Local mode failed, fall back to web
        logger.info("→ FALLBACK: Local failed, using web search")
        return "web_search"
    else:
        logger.info("→ LOCAL MODE: Using similarity-based retrieval")
        return "retrieve"  # Go to retrieval (will use similar docs or hybrid retrieval)

def route_after_grading(state: GraphState) -> str:
    """
    Route based on document grading confidence and search mode
    """
    search_mode = state.get("search_mode", "local")
    web_search_needed = state.get("web_search_needed", False)
    query_rewrite_needed = state.get("query_rewrite_needed", False)

    # For comparison mode, might need to get the other result type
    if search_mode == "compare":
        local_generation = state.get("local_generation")
        web_generation = state.get("web_generation")

        logger.info(f"COMPARE MODE routing - Local: {bool(local_generation)}, Web: {bool(web_generation)}, Web Needed: {web_search_needed}")

        if web_search_needed and not web_generation:
            # Need web results for comparison
            logger.info("→ COMPARE MODE: Need web results for comparison")
            return "web_search"
        elif query_rewrite_needed:
            logger.info("→ COMPARE MODE: Query rewrite needed")
            return "generate"
        else:
            logger.info("→ COMPARE MODE: Generating result")
            return "generate"

    # For other modes, standard routing
    if web_search_needed:
        logger.info("→ WEB SEARCH NEEDED")
        return "web_search"
    elif query_rewrite_needed:
        logger.info("→ QUERY REWRITE NEEDED")
        return "generate"  # Skip reoptimization, go directly to generation
    else:
        logger.info("→ PROCEED TO GENERATION")
        return "generate"

def route_after_generation(state: GraphState) -> str:
    """
    Route after generation based on search mode
    """
    search_mode = state.get("search_mode", "local")
    local_generation = state.get("local_generation")
    web_generation = state.get("web_generation")

    logger.info(f"Post-generation routing - Mode: {search_mode}, Local: {bool(local_generation)}, Web: {bool(web_generation)}")

    if search_mode == "compare":
        # For comparison mode, check if we have both results
        if local_generation and not web_generation:
            # Have local result, need web result
            logger.info("→ COMPARE MODE: Have local result, need web result")
            return "web_search"
        elif web_generation and not local_generation:
            # Have web result, need local result
            logger.info("→ COMPARE MODE: Have web result, need local result")
            return "retrieve"
        else:
            # Have both or neither, proceed to end
            logger.info("→ COMPARE MODE: Analysis complete, proceeding to end")
            return "__end__"

    # For other modes, we're done
    logger.info("→ WORKFLOW COMPLETE")
    return "__end__"


# ==================== Build LangGraph Workflow ====================

logger.info("Building Context Engineered RAG workflow...")

# Initialize state graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("analyze_context", analyze_query_context)
workflow.add_node("check_similarity", check_document_similarity)
workflow.add_node("retrieve", hybrid_retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate_with_context)

# Build graph edges
workflow.add_edge(START, "analyze_context")
workflow.add_edge("analyze_context", "check_similarity")

# Conditional routing after similarity check
workflow.add_conditional_edges(
    "check_similarity",
    route_after_similarity_check,
    {
        "retrieve": "retrieve",
        "web_search": "web_search"
    }
)

workflow.add_edge("retrieve", "grade_documents")

# Conditional routing after grading
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

# After web search, go to generation
workflow.add_edge("web_search", "generate")

# Conditional routing after generation for dual generation support
workflow.add_conditional_edges(
    "generate",
    route_after_generation,
    {
        "web_search": "web_search",
        "retrieve": "retrieve",
        "__end__": END
    }
)

# Compile workflow
app = workflow.compile()

logger.info("✓ Context Engineered RAG workflow compiled successfully")

# ==================== Main Execution ====================

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("SIMPLIFIED CONTEXT ENGINEERED RAG SYSTEM - READY")
    logger.info("="*80 + "\n")

    # Test query
    test_query = "สร้างหนังสือเชิญประชุมกับคณะกรรมการ ITU เรื่องการป้องกันแก๊ง call center ในประเทศเพื่อนบ้าน โดยจะเสนอมาตรฐานการลดกำลังส่ง cellsite กำหนดวันที่ 10 กรกฎาคม 2568 เวลา 09:00 น. ณ ห้องประชุมใหญ่ สำนักงาน กสทช"

    input_state = {
        "question": test_query,
        "web_search_needed": False,
        "query_rewrite_needed": False,
        "regeneration_needed": False,
        "warnings": []
    }

    logger.info(f"Processing query: {test_query[:50]}...\n")

    # Configure execution
    run_config = {
        "configurable": {"thread_id": "simple_test"},
        "recursion_limit": 50
    }

    # Execute workflow
    final_state = None
    last_generation = None

    for output in app.stream(input_state, run_config):
        for node_name, node_output in output.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Node: '{node_name}' completed")
            if node_name == "analyze_context":
                logger.info(f"  Document Type: {node_output.get('document_type')}")
                logger.info(f"  Extracted Info: {len(node_output.get('extracted_info', {}))} categories")
            elif node_name == "generate":
                logger.info(f"  Generated Length: {len(node_output.get('generation', ''))}")
            logger.info(f"{'='*60}")

            if 'generation' in node_output:
                last_generation = node_output['generation']
            final_state = node_output

    # Print final generation
    print("\n" + "="*80)
    print("FINAL GENERATED DOCUMENT")
    print("="*80 + "\n")

    if last_generation:
        print(last_generation)
    elif final_state and 'generation' in final_state:
        print(final_state['generation'])
    else:
        print("(No generation found)")

    # Print detailed results
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

        # Thai Format Validation
        if final_state.get('generation'):
            generation = final_state['generation']
            thai_validation = validate_thai_document(generation)
            print(f"\n🇹🇭 Thai Format Validation:")
            print(f"   Compliance Score: {thai_validation['overall_score']:.2f}")
            print(f"   Is Compliant: {'✅' if thai_validation['is_compliant'] else '❌'}")
            if thai_validation['issues']:
                print(f"   Issues: {', '.join(thai_validation['issues'])}")

    print("\n" + "="*80)