"""
Advanced Agentic RAG System for Thai Government Document Generation
Implements CRAG with Hybrid Retrieval, Reranking, and Self-Reflection

Features:
- Hybrid Retrieval (BM25 + Dense Vectors + Reranking)
- Query Optimization with rewriting and decomposition
- Corrective RAG with confidence-based routing
- Self-Reflection for quality validation
- RAGAS evaluation framework
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI  # Using OpenAI-compatible API for DeepSeek
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.pregel.retry import RetryPolicy

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated
from typing_extensions import TypedDict
import operator
from dotenv import load_dotenv
from pprint import pprint
import logging

# Import configuration
import config as app_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Validate configuration
app_config.validate_config()

# ==================== Pydantic Models for Structured Output ====================

class DocumentGrade(BaseModel):
    """Document relevance grade with confidence score"""
    confidence_score: float = Field(
        description="คะแนนความเชื่อมั่นจาก 0.0 ถึง 1.0 ที่บ่งบอกความเกี่ยวข้องของเอกสาร",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="เหตุผลในการให้คะแนน อธิบายสั้นๆ ว่าทำไมถึงให้คะแนนนี้"
    )
    key_points: List[str] = Field(
        description="จุดสำคัญที่เกี่ยวข้องกับคำถาม (ถ้ามี)"
    )


class OptimizedQuery(BaseModel):
    """Optimized query for better retrieval"""
    rewritten_query: str = Field(
        description="คำค้นหาที่ปรับปรุงแล้วให้ชัดเจนและมีประสิทธิภาพมากขึ้น"
    )
    additional_keywords: List[str] = Field(
        description="คำสำคัญเพิ่มเติมที่เกี่ยวข้อง"
    )
    query_type: str = Field(
        description="ประเภทของคำค้นหา เช่น 'ขออนุมัติ', 'เชิญประชุม', 'รายงาน'"
    )


class QualityAssessment(BaseModel):
    """Quality assessment of generated document"""
    overall_score: float = Field(
        description="คะแนนรวมความคุณภาพ 0.0-1.0",
        ge=0.0,
        le=1.0
    )
    format_correctness: float = Field(
        description="คะแนนความถูกต้องของรูปแบบ 0.0-1.0",
        ge=0.0,
        le=1.0
    )
    content_accuracy: float = Field(
        description="คะแนนความถูกต้องของเนื้อหา 0.0-1.0",
        ge=0.0,
        le=1.0
    )
    language_quality: float = Field(
        description="คะแนนคุณภาพภาษา 0.0-1.0",
        ge=0.0,
        le=1.0
    )
    completeness: float = Field(
        description="คะแนนความสมบูรณ์ 0.0-1.0",
        ge=0.0,
        le=1.0
    )
    strengths: List[str] = Field(
        description="จุดแข็งของเอกสาร"
    )
    improvements: List[str] = Field(
        description="จุดที่ควรปรับปรุง"
    )
    recommendations: Optional[str] = Field(
        default=None,
        description="คำแนะนำในการปรับปรุง (ถ้าคะแนน < 0.8)"
    )


# ==================== Enhanced State Schema ====================

class GraphState(TypedDict):
    """
    Enhanced state schema for advanced agentic RAG system
    """
    # Original query
    question: str

    # Query optimization
    optimized_query: Optional[str]
    query_type: Optional[str]
    additional_keywords: Optional[List[str]]

    # Retrieval
    documents: List[Document]
    filtered_docs: List[Document]
    reranked_docs: List[Document]

    # Document grading
    doc_grades: List[Dict]  # List of {doc_id, confidence, reasoning, key_points}
    average_confidence: Optional[float]

    # Generation
    generation: Optional[str]

    # Self-reflection
    quality_score: Optional[float]
    quality_assessment: Optional[Dict]
    reflection_iteration: Annotated[int, operator.add]

    # Control flow
    web_search_needed: bool
    query_rewrite_needed: bool
    regeneration_needed: bool

    # Metadata
    retrieval_method: Optional[str]  # "hybrid", "dense", "bm25"
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
    weights=[0.5, 0.5]  # Equal weighting
)

# 4. Reranker
reranker = CohereRerank(
    model=app_config.RERANK_MODEL,
    top_n=app_config.RERANK_TOP_K
)

logger.info("✓ Hybrid retrieval system ready")

# ==================== LLM Setup ====================

# Generation LLM (DeepSeek - better quality than Typhoon 3B)
llm_generation = ChatOpenAI(
    model=app_config.GENERATION_MODEL,
    base_url="https://api.deepseek.com",
    api_key=app_config.DEEPSEEK_API_KEY,
    temperature=app_config.GENERATION_TEMPERATURE
)

# Grading and Optimization LLM (DeepSeek via OpenAI-compatible API)
# Note: DeepSeek doesn't support structured output, so we'll use JSON mode
llm_small = ChatOpenAI(
    model=app_config.GRADING_MODEL,
    base_url="https://api.deepseek.com",
    api_key=app_config.DEEPSEEK_API_KEY,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}  # Use JSON mode instead of structured output
)

# ==================== Web Search Tool ====================

web_search_tool = TavilySearchResults(k=app_config.WEB_SEARCH_RESULTS)

# ==================== Agent 1: Query Optimization ====================

def optimize_query(state: GraphState) -> GraphState:
    """
    Optimize the user query for better retrieval
    - Rewrite query for clarity
    - Add relevant keywords
    - Identify query type
    """
    logger.info("--- QUERY OPTIMIZATION ---")

    question = state["question"]

    # Use JSON mode instead of structured output for DeepSeek
    prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.QUERY_OPTIMIZER_SYSTEM_PROMPT + "\n\nตอบเป็น JSON format: {{\"rewritten_query\": \"...\", \"additional_keywords\": [...], \"query_type\": \"...\"}}"),
        ("human", "คำถามต้นฉบับ: {question}\n\nกรุณาปรับปรุงคำค้นหาให้ดีขึ้น")
    ])

    try:
        optimization_chain = prompt | llm_small | StrOutputParser()
        result_str = optimization_chain.invoke({"question": question})

        # Parse JSON response
        import json
        result_json = json.loads(result_str)

        # Create OptimizedQuery object from JSON
        result = OptimizedQuery(
            rewritten_query=result_json.get("rewritten_query", question),
            additional_keywords=result_json.get("additional_keywords", []),
            query_type=result_json.get("query_type", "general")
        )

        logger.info(f"Original query: {question}")
        logger.info(f"Optimized query: {result.rewritten_query}")
        logger.info(f"Query type: {result.query_type}")
        logger.info(f"Additional keywords: {result.additional_keywords}")

        return {
            "optimized_query": result.rewritten_query,
            "query_type": result.query_type,
            "additional_keywords": result.additional_keywords
        }

    except Exception as e:
        logger.error(f"Query optimization failed: {e}")
        return {
            "optimized_query": question,
            "query_type": "general",
            "additional_keywords": [],
            "warnings": [f"Query optimization failed: {str(e)}"]
        }


# ==================== Agent 2: Hybrid Retrieval ====================

def hybrid_retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents using hybrid approach:
    1. BM25 (sparse) + Dense vectors
    2. Rerank with Cohere
    """
    logger.info("--- HYBRID RETRIEVAL ---")

    # Use optimized query if available, otherwise original
    query = state.get("optimized_query") or state["question"]

    try:
        # Step 1: Ensemble retrieval (BM25 + Dense)
        logger.info("Retrieving with ensemble (BM25 + Dense)...")
        ensemble_docs = ensemble_retriever.invoke(query)
        logger.info(f"Retrieved {len(ensemble_docs)} documents from ensemble")

        # Step 2: Rerank with Cohere
        logger.info("Reranking documents...")

        # Prepare documents for reranking
        docs_for_rerank = [doc.page_content for doc in ensemble_docs]

        # Rerank
        reranked_results = reranker.rerank(
            query=query,
            documents=docs_for_rerank
        )

        # Get reranked documents (Cohere v2 API returns dict, not object)
        if hasattr(reranked_results, 'results'):
            # Cohere client response object
            reranked_docs = [ensemble_docs[result['index']] for result in reranked_results.results]
        elif isinstance(reranked_results, list):
            # Direct list response
            reranked_docs = [ensemble_docs[result['index']] for result in reranked_results]
        else:
            # Fallback: assume it's iterable with index attribute
            reranked_docs = [ensemble_docs[result.index if hasattr(result, 'index') else result['index']] for result in reranked_results]

        logger.info(f"Reranked to top {len(reranked_docs)} documents")

        return {
            "documents": ensemble_docs,
            "reranked_docs": reranked_docs,
            "retrieval_method": "hybrid_reranked"
        }

    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        # Fallback to dense retrieval only
        logger.info("Falling back to dense retrieval only...")

        try:
            docs = dense_retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents from dense retrieval")

            return {
                "documents": docs,
                "reranked_docs": docs[:app_config.RERANK_TOP_K],
                "retrieval_method": "dense_fallback",
                "warnings": [f"Hybrid retrieval failed, used dense fallback: {str(e)}"]
            }

        except Exception as e2:
            logger.error(f"Even dense retrieval failed: {e2}")
            return {
                "documents": [],
                "reranked_docs": [],
                "retrieval_method": "failed",
                "error": f"Retrieval failed: {str(e2)}"
            }


# ==================== Agent 3: Document Grader (CRAG) ====================

def grade_documents(state: GraphState) -> GraphState:
    """
    Grade document relevance with confidence scores
    Implements Corrective RAG pattern
    """
    logger.info("--- DOCUMENT GRADING (CRAG) ---")

    question = state.get("optimized_query") or state["question"]
    documents = state.get("reranked_docs", [])

    if not documents:
        logger.warning("No documents to grade")
        return {
            "filtered_docs": [],
            "doc_grades": [],
            "average_confidence": 0.0,
            "web_search_needed": True,
            "query_rewrite_needed": False
        }

    # Use JSON mode for DeepSeek
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.DOCUMENT_GRADER_SYSTEM_PROMPT + "\n\nตอบเป็น JSON format: {{\"confidence_score\": 0.0-1.0, \"reasoning\": \"...\", \"key_points\": [...]}}"),
        ("human", "เอกสาร:\n\n{document}\n\nคำถาม: {question}\n\nกรุณาประเมินความเกี่ยวข้อง")
    ])

    grader_chain = grade_prompt | llm_small | StrOutputParser()

    # Grade each document
    doc_grades = []
    filtered_docs = []

    for i, doc in enumerate(documents[:app_config.FINAL_CONTEXT_K]):
        try:
            grade_str = grader_chain.invoke({
                "document": doc.page_content[:1500],  # Limit content for grading
                "question": question
            })

            # Parse JSON response
            import json
            grade_json = json.loads(grade_str)

            # Create DocumentGrade object
            grade = DocumentGrade(
                confidence_score=float(grade_json.get("confidence_score", 0.5)),
                reasoning=grade_json.get("reasoning", ""),
                key_points=grade_json.get("key_points", [])
            )

            doc_grades.append({
                "doc_index": i,
                "confidence": grade.confidence_score,
                "reasoning": grade.reasoning,
                "key_points": grade.key_points
            })

            logger.info(f"Doc {i+1}: Confidence={grade.confidence_score:.2f}, Reasoning={grade.reasoning[:100]}...")

            # Filter documents above minimum threshold
            if grade.confidence_score >= app_config.LOW_CONFIDENCE_THRESHOLD:
                filtered_docs.append(doc)

        except Exception as e:
            logger.error(f"Failed to grade document {i}: {e}")
            doc_grades.append({
                "doc_index": i,
                "confidence": 0.5,  # Neutral score on error
                "reasoning": f"Grading failed: {str(e)}",
                "key_points": []
            })

    # Calculate average confidence
    if doc_grades:
        average_confidence = sum(g["confidence"] for g in doc_grades) / len(doc_grades)
    else:
        average_confidence = 0.0

    logger.info(f"Average confidence: {average_confidence:.2f}")
    logger.info(f"Filtered docs: {len(filtered_docs)} out of {len(documents)}")

    # Determine routing based on confidence
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

    return {
        "filtered_docs": filtered_docs,
        "doc_grades": doc_grades,
        "average_confidence": average_confidence,
        "web_search_needed": web_search_needed,
        "query_rewrite_needed": query_rewrite_needed
    }


# ==================== Agent 4: Web Search ====================

def web_search(state: GraphState) -> GraphState:
    """
    Perform web search when document confidence is very low
    """
    logger.info("--- WEB SEARCH ---")

    question = state.get("optimized_query") or state["question"]

    try:
        # Prepare search query
        search_prompt = f"หนังสือราชการไทย {question}"

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

        return {
            "filtered_docs": combined_docs,
            "web_search_needed": False  # Reset flag
        }

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "web_search_needed": False,
            "warnings": [f"Web search failed: {str(e)}"]
        }


# ==================== Agent 5: Generator ====================

def generate(state: GraphState) -> GraphState:
    """
    Generate Thai government document based on retrieved context
    """
    logger.info("--- GENERATION ---")

    question = state["question"]
    documents = state.get("filtered_docs", [])

    if not documents:
        logger.warning("No documents available for generation")
        return {
            "generation": "ไม่สามารถสร้างเอกสารได้เนื่องจากไม่พบข้อมูลที่เกี่ยวข้อง กรุณาระบุรายละเอียดเพิ่มเติม",
            "quality_score": 0.0,
            "regeneration_needed": True
        }

    # Format context
    context = "\n\n---\n\n".join([doc.page_content for doc in documents[:app_config.FINAL_CONTEXT_K]])

    # Create generation prompt
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.GENERATION_SYSTEM_PROMPT),
        ("human", "ตัวอย่างบันทึกข้อความและข้อมูลอ้างอิง:\n\n{context}\n\nคำขอ: {question}\n\nกรุณาสร้างหนังสือราชการตามโครงสร้างที่กำหนด")
    ])

    try:
        rag_chain = generation_prompt | llm_generation | StrOutputParser()
        generation = rag_chain.invoke({"context": context, "question": question})

        logger.info(f"Generated document ({len(generation)} chars)")

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


# ==================== Agent 6: Self-Reflection (Quality Validator) ====================

def reflect_on_generation(state: GraphState) -> GraphState:
    """
    Validate quality of generated document
    Trigger regeneration if quality is below threshold
    """
    logger.info("--- SELF-REFLECTION ---")

    generation = state.get("generation")
    question = state["question"]
    documents = state.get("filtered_docs", [])
    iteration = state.get("reflection_iteration", 0)

    if not generation:
        return {
            "quality_score": 0.0,
            "regeneration_needed": True
        }

    # Create context summary
    context_summary = "\n".join([doc.page_content[:300] for doc in documents[:3]])

    # Use JSON mode for DeepSeek
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.REFLECTION_SYSTEM_PROMPT + "\n\nตอบเป็น JSON format: {{\"overall_score\": 0.0-1.0, \"format_correctness\": 0.0-1.0, \"content_accuracy\": 0.0-1.0, \"language_quality\": 0.0-1.0, \"completeness\": 0.0-1.0, \"strengths\": [...], \"improvements\": [...], \"recommendations\": \"...\"}}"),
        ("human", """คำขอต้นฉบับ: {question}

เอกสารที่สร้างขึ้น:
{generation}

ข้อมูลอ้างอิง (สรุป):
{context}

กรุณาประเมินคุณภาพของเอกสาร""")
    ])

    try:
        reflection_chain = reflection_prompt | llm_small | StrOutputParser()
        assessment_str = reflection_chain.invoke({
            "question": question,
            "generation": generation,
            "context": context_summary
        })

        # Parse JSON response
        import json
        assessment_json = json.loads(assessment_str)

        # Create QualityAssessment object
        assessment = QualityAssessment(
            overall_score=float(assessment_json.get("overall_score", 0.7)),
            format_correctness=float(assessment_json.get("format_correctness", 0.7)),
            content_accuracy=float(assessment_json.get("content_accuracy", 0.7)),
            language_quality=float(assessment_json.get("language_quality", 0.7)),
            completeness=float(assessment_json.get("completeness", 0.7)),
            strengths=assessment_json.get("strengths", []),
            improvements=assessment_json.get("improvements", []),
            recommendations=assessment_json.get("recommendations")
        )

        logger.info(f"Quality Assessment:")
        logger.info(f"  Overall: {assessment.overall_score:.2f}")
        logger.info(f"  Format: {assessment.format_correctness:.2f}")
        logger.info(f"  Content: {assessment.content_accuracy:.2f}")
        logger.info(f"  Language: {assessment.language_quality:.2f}")
        logger.info(f"  Completeness: {assessment.completeness:.2f}")
        logger.info(f"  Strengths: {', '.join(assessment.strengths)}")

        if assessment.improvements:
            logger.info(f"  Improvements needed: {', '.join(assessment.improvements)}")

        # Determine if regeneration is needed
        regeneration_needed = False

        if assessment.overall_score < app_config.REFLECTION_THRESHOLD:
            if iteration < app_config.MAX_REFLECTION_ITERATIONS:
                logger.info(f"→ Quality below threshold ({assessment.overall_score:.2f} < {app_config.REFLECTION_THRESHOLD})")
                logger.info(f"→ Triggering REGENERATION (iteration {iteration + 1}/{app_config.MAX_REFLECTION_ITERATIONS})")
                regeneration_needed = True
            else:
                logger.warning(f"→ Max iterations reached, accepting current quality")
        else:
            logger.info(f"→ Quality acceptable ({assessment.overall_score:.2f} >= {app_config.REFLECTION_THRESHOLD})")

        return {
            "quality_score": assessment.overall_score,
            "quality_assessment": {
                "format_correctness": assessment.format_correctness,
                "content_accuracy": assessment.content_accuracy,
                "language_quality": assessment.language_quality,
                "completeness": assessment.completeness,
                "strengths": assessment.strengths,
                "improvements": assessment.improvements,
                "recommendations": assessment.recommendations
            },
            "regeneration_needed": regeneration_needed,
            "reflection_iteration": 1  # Will be added to existing count
        }

    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        # On reflection error, accept the generation
        return {
            "quality_score": 0.7,  # Neutral score
            "regeneration_needed": False,
            "warnings": [f"Reflection failed: {str(e)}"]
        }


# ==================== Routing Functions ====================

def route_after_grading(state: GraphState) -> str:
    """
    Route based on document grading confidence
    """
    web_search_needed = state.get("web_search_needed", False)
    query_rewrite_needed = state.get("query_rewrite_needed", False)

    if web_search_needed:
        return "web_search"
    elif query_rewrite_needed:
        return "reoptimize_query"
    else:
        return "generate"


def route_after_reflection(state: GraphState) -> str:
    """
    Route based on quality assessment
    """
    regeneration_needed = state.get("regeneration_needed", False)
    iteration = state.get("reflection_iteration", 0)

    if regeneration_needed and iteration < app_config.MAX_REFLECTION_ITERATIONS:
        return "generate"
    else:
        return END


# ==================== Build LangGraph Workflow ====================

logger.info("Building LangGraph workflow...")

# Initialize state graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("optimize_query", optimize_query)
workflow.add_node("retrieve", hybrid_retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("reoptimize_query", optimize_query)  # Reuse optimizer for rewriting
workflow.add_node("generate", generate)
workflow.add_node("reflect", reflect_on_generation)

# Build graph edges
workflow.add_edge(START, "optimize_query")
workflow.add_edge("optimize_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional routing after grading
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "web_search": "web_search",
        "reoptimize_query": "reoptimize_query",
        "generate": "generate"
    }
)

# After web search, go to generation
workflow.add_edge("web_search", "generate")

# After query rewrite, retry retrieval
workflow.add_edge("reoptimize_query", "retrieve")

# After generation, reflect
workflow.add_edge("generate", "reflect")

# Conditional routing after reflection
workflow.add_conditional_edges(
    "reflect",
    route_after_reflection,
    {
        "generate": "generate",
        END: END
    }
)

# Compile with checkpointing
if app_config.ENABLE_CHECKPOINTING:
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
else:
    app = workflow.compile()

logger.info("✓ LangGraph workflow compiled successfully")

# ==================== Main Execution ====================

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("ADVANCED AGENTIC RAG SYSTEM - READY")
    logger.info("="*80 + "\n")

    # Test queries
    input_state_1 = {
        "question": "เขียนหนังสือบันทึกข้อความ ขออนุมัติเดินทางตรวจตามแผน",
        "reflection_iteration": 0,
        "web_search_needed": False,
        "query_rewrite_needed": False,
        "regeneration_needed": False,
        "warnings": []
    }

    input_state_2 = {
        "question": "สร้างหนังสือราชการเชิญประชุมกับคณะกรรมการ ITU โดยมีรายละเอียด การป้องกัน แก๊ง call center ในประเทศเพื่อนบ้าน โดยจะมีการเสนอมาตรฐานการป้องกันแก๊ง call center ลดกำลังส่ง cellsite เพื่อไม่ให้ใช้สัญญาณ internet ข้ามไปยังประเทศเพื่อนบ้าน",
        "reflection_iteration": 0,
        "web_search_needed": False,
        "query_rewrite_needed": False,
        "regeneration_needed": False,
        "warnings": []
    }

    input_state_3 = {
        "question": "สร้างหนังสือราชการบันทึกข้อความขออนุมัติเดินทางตรวจสอบตามแผนประจำปี 2568 โดยมีข้อมูล เรื่องตรวจสอบตามแผน อ้างอิง สทช2203/144 เดินทางจังหวัดบุรีรัมย์ 1 มกราคม - 2 มกราคม 2568",
        "reflection_iteration": 0,
        "web_search_needed": False,
        "query_rewrite_needed": False,
        "regeneration_needed": False,
        "warnings": []
    }

    # Choose test case
    test_input = input_state_2

    logger.info(f"Processing query: {test_input['question']}\n")

    # Configure execution with thread_id for checkpointing and recursion limit
    run_config = {
        "configurable": {"thread_id": "test_session_1"},
        "recursion_limit": 50  # Increase from default 25 to handle retries
    }

    # Stream execution and collect states
    final_state = None
    last_generation = None
    for output in app.stream(test_input, run_config):
        for node_name, node_output in output.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Node: '{node_name}' completed")
            logger.info(f"{'='*60}")
            # Keep track of the last generation
            if 'generation' in node_output:
                last_generation = node_output['generation']
            # Always update final_state
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

    # Print quality assessment
    if final_state and 'quality_assessment' in final_state:
        print("\n" + "="*80)
        print("QUALITY ASSESSMENT")
        print("="*80)
        qa = final_state['quality_assessment']
        print(f"\nOverall Score: {final_state.get('quality_score', 0):.2f}")
        print(f"Format Correctness: {qa.get('format_correctness', 0):.2f}")
        print(f"Content Accuracy: {qa.get('content_accuracy', 0):.2f}")
        print(f"Language Quality: {qa.get('language_quality', 0):.2f}")
        print(f"Completeness: {qa.get('completeness', 0):.2f}")
        print(f"\nStrengths: {', '.join(qa.get('strengths', []))}")
        if qa.get('improvements'):
            print(f"Improvements: {', '.join(qa.get('improvements', []))}")
        if qa.get('recommendations'):
            print(f"\nRecommendations: {qa.get('recommendations')}")

    # Print warnings if any
    if final_state and 'warnings' in final_state and final_state['warnings']:
        print("\n" + "="*80)
        print("WARNINGS")
        print("="*80)
        for warning in final_state['warnings']:
            print(f"⚠ {warning}")

    print("\n" + "="*80)