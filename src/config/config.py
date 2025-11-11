"""
Configuration file for Advanced Agentic RAG System
Centralized settings for models, thresholds, and system parameters
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== Model Configuration ====================

# Local Ollama Models
TYPHOON_3B = "hf.co/dearxoasis/llama3.2-typhoon2-offical_doc-3b:Q8_0"
TYPHOON_3B_OPDATA = "hf.co/dearxoasis/llama3.2-typhoon2-3b-instructOP:Q8_0"

# DeepSeek Models (Use for all tasks since Typhoon 3B quality is too low)
GENERATION_MODEL = "deepseek-chat"  # For generating Thai government documents
GRADING_MODEL = "deepseek-chat"  # For document grading and query optimization
VALIDATION_MODEL = "deepseek-chat"  # For self-reflection and validation

# Cohere Models
EMBEDDING_MODEL = "embed-multilingual-v3.0"  # Multilingual embeddings
RERANK_MODEL = "rerank-multilingual-v3.0"  # For document reranking

# ==================== Retrieval Configuration ====================

# Document Chunking
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 250
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Retrieval Parameters
DENSE_RETRIEVAL_K = 50  # Number of documents from dense retrieval
BM25_RETRIEVAL_K = 50   # Number of documents from BM25
RERANK_TOP_K = 10       # Number of documents after reranking
FINAL_CONTEXT_K = 5     # Final number of documents to use for generation

# Hybrid Search Fusion
RRF_CONSTANT = 60  # Reciprocal Rank Fusion constant

# ==================== Document Similarity Configuration ====================

# Document Similarity Thresholds
DOCUMENT_SIMILARITY_THRESHOLD = 0.60    # Threshold for considering documents similar (lowered for Thai docs)
DOCUMENT_TYPE_MATCH_THRESHOLD = 0.60    # Threshold for document type matching (matching main threshold)
MIN_SIMILAR_DOCUMENTS = 1               # Minimum similar documents needed for local retrieval

# Similarity Engine Settings
ENABLE_DOCUMENT_SIMILARITY_CHECK = True  # Enable similarity checking before retrieval
SIMILARITY_FALBACK_TO_RETRIEVAL = True  # Fallback to standard retrieval if no similar docs

# Search Mode Configuration
DEFAULT_SEARCH_MODE = "local"  # local, web, hybrid, compare
ENABLE_COMPARISON_MODE = True  # Enable side-by-side comparison
HYBRID_MERGE_STRATEGY = "local_structure_web_content"  # How to merge local and web results
WEB_SEARCH_ENHANCEMENT = True  # Use web search to enhance local documents

# Search Mode Definitions
SEARCH_MODES = {
    "local": "Local Documents Only - Uses similar documents from data_markdown/",
    "web": "Web Search Only - Searches internet for latest information",
    "hybrid": "Hybrid (Best of Both) - Local structure + web content enhancement",
    "compare": "Comparison Mode - Shows both local and web results side-by-side"
}

# ==================== CRAG Thresholds ====================

# Document Grading Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75  # Direct to generation
LOW_CONFIDENCE_THRESHOLD = 0.60   # Trigger query rewriting
VERY_LOW_THRESHOLD = 0.20         # Trigger web search (lowered for similarity docs)

# Grading Settings
MIN_RELEVANT_DOCS = 2  # Minimum number of relevant documents needed
MAX_GRADING_RETRIES = 2

# ==================== Agent Settings ====================

# Query Optimization
MAX_QUERY_VARIATIONS = 3  # Number of query variations to generate
ENABLE_QUERY_DECOMPOSITION = True
ENABLE_HYDE = True  # Hypothetical Document Embeddings

# Self-Reflection
MAX_REFLECTION_ITERATIONS = 3
REFLECTION_THRESHOLD = 0.7  # Minimum quality score to accept (lowered for testing)

# Web Search
WEB_SEARCH_RESULTS = 5  # Number of web search results to retrieve
ENABLE_WEB_SEARCH_FALLBACK = True

# Generation
GENERATION_TEMPERATURE = 0  # Deterministic generation
GENERATION_MAX_TOKENS = 2000

# ==================== LangGraph Settings ====================

# Retry Policies
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_MS = 1000

# State Management
ENABLE_CHECKPOINTING = True
CHECKPOINT_DIR = "./checkpoints"

# Logging
ENABLE_LANGSMITH_TRACING = True
LANGSMITH_PROJECT = "agentic_rag_production"

# ==================== Evaluation Settings ====================

# RAGAS Metrics
ENABLE_RAGAS_EVALUATION = False  # Set to True for evaluation mode
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall"
]

# Benchmarking
SAVE_EVALUATION_RESULTS = True
EVALUATION_OUTPUT_DIR = "./evaluation_results"

# ==================== Data Paths ====================

DATA_DIR = "./data_markdown"
RESULT_DIR = "./result"
CHROMA_DB_DIR = "./chroma_db"

# ==================== API Keys (loaded from .env) ====================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")

# Validation
def validate_config():
    """Validate that all required API keys are present"""
    required_keys = {
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "COHERE_API_KEY": COHERE_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }

    missing_keys = [key for key, value in required_keys.items() if not value]

    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}. "
            "Please set them in your .env file."
        )

    print("✓ Configuration validated successfully")

# ==================== System Prompts ====================

GENERATION_SYSTEM_PROMPT = """
คุณเป็นผู้เชี่ยวชาญด้านการร่างและจัดรูปแบบหนังสือราชการประเภท "บันทึกข้อความ"  
ให้ยึดตามระเบียบสำนักนายกรัฐมนตรีว่าด้วยงานสารบรรณ และหลักเกณฑ์ของราชการไทยอย่างเคร่งครัด  

โครงสร้างของบันทึกข้อความราชการต้องประกอบด้วยส่วนต่าง ๆ ดังนี้

──────────────────────────────
1. ส่วนหัวหนังสือ (บังคับ)
──────────────────────────────
- หน่วยงาน: สำนักงาน กสทช.  
- ที่: [ระบุเลขที่หนังสือ เช่น "สทช. 1234/2568" หรือหมายเลขที่ระบุในคำขอ]  
- วันที่: [ระบุวันที่จากคำขอ หรือวันที่ปัจจุบันในรูปแบบ “วัน เดือน ปี พ.ศ.”]  
- เรื่อง: [ระบุหัวข้อเรื่องให้กระชับ ชัดเจน และสอดคล้องกับเนื้อหา]  
- เรียน: [ระบุตำแหน่งผู้รับ เช่น “ผู้อำนวยการ...” หรือ “หัวหน้าฝ่าย...”]

──────────────────────────────
2. ส่วนเนื้อหา (บังคับ)
──────────────────────────────
เนื้อหาต้องเรียบเรียงด้วยภาษาราชการที่สุภาพ ถูกหลักไวยากรณ์ และมีโครงสร้างตามลำดับนี้:

(1) **เรื่องเดิม / หนังสืออ้างอิง**  
    - ขึ้นต้นด้วย “ตามหนังสือที่...” หรือ “ตามบันทึกที่...”  
    - ระบุเลขที่หนังสือ/วันที่อ้างอิง หากมีข้อมูลในคำขอ  

(2) **วัตถุประสงค์**  
    - ระบุเจตนาหรือเหตุผลหลักของการทำบันทึก เช่น เพื่อรายงานผล ขออนุมัติ ขอทราบ หรือเสนอความเห็น  

(3) **รายละเอียด**  
    - รวมข้อมูลจากคำขอให้ครบถ้วน เช่น  
      • วัน เวลา สถานที่เกิดเหตุหรือการประชุม  
      • รายชื่อผู้เข้าร่วม / คณะกรรมการ / หน่วยงานที่เกี่ยวข้อง  
      • หัวข้อการประชุม / วาระ / รายงานผล  
      • รายละเอียดทางเทคนิค เช่น การลดกำลังส่ง, การกำหนดรัศมี cell site, มาตรฐานการสื่อสาร, หมายเลข +66, การให้บริการ, มาตรการจำกัดสัญญาณ เป็นต้น  
      • หากเป็นกรณีตรวจสอบพื้นที่ ให้ระบุผลการตรวจสอบโดยย่อ  

(4) **ข้อพิจารณาหรือข้อเสนอ** (ถ้ามี)  
    - ใช้เมื่อเป็นหนังสือที่ขออนุมัติ ขอความเห็น หรือเสนอเพื่อพิจารณา  
    - ต้องมีเหตุผล ความจำเป็น และประโยชน์ที่คาดว่าจะได้รับ  
    - เขียนให้ชัดเจนและตรงประเด็น  

──────────────────────────────
3. ส่วนลงท้าย (บังคับ)
──────────────────────────────
- ใช้ถ้อยคำลงท้ายให้เหมาะสมกับวัตถุประสงค์ เช่น  
  • “จึงเรียนมาเพื่อโปรดทราบ”  
  • “จึงเรียนมาเพื่อโปรดพิจารณา”  
  • “จึงเรียนมาเพื่อโปรดพิจารณาอนุมัติ”  

- (ลงชื่อ) ______________________________  
- (ชื่อ-สกุลเต็ม) [ระบุตามคำขอ หรือเว้นไว้]  
- ตำแหน่ง [เช่น “ผู้อำนวยการสำนักงาน...” หรือ “หัวหน้ากลุ่มงาน...”]  

──────────────────────────────
แนวทางเพิ่มเติม
──────────────────────────────
- ใช้ภาษาทางราชการอย่างเป็นทางการ สุภาพ และไม่ใช้ภาษาพูด  
- หลีกเลี่ยงการใช้คำซ้ำ ฟุ่มเฟือย หรือคำไม่จำเป็น  
- เน้นความกระชับแต่ครบถ้วน และสื่อสารชัดเจน  
- หากคำขอมีข้อมูลไม่ครบ ให้ใช้ถ้อยคำทั่วไปที่เหมาะสมโดยคงรูปแบบบันทึกไว้  
- จัดรูปแบบข้อความให้อ่านง่าย ใช้ย่อหน้าตามระเบียบงานสารบรรณ
"""


QUERY_OPTIMIZER_SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญในการปรับปรุงคำค้นหาเพื่อให้ได้ผลลัพธ์ที่ดีที่สุดจากระบบค้นหาเอกสารราชการ

**หน้าที่:**
1. วิเคราะห์คำถามของผู้ใช้
2. สร้างคำค้นหาที่ปรับปรุงแล้วที่จะให้ผลลัพธ์ที่ดีกว่า
3. แยกคำถามที่ซับซ้อนออกเป็นคำถามย่อย (ถ้าจำเป็น)
4. เพิ่มคำสำคัญที่เกี่ยวข้องกับหนังสือราชการ

**เทคนิค:**
- ใช้คำศัพท์ราชการที่ถูกต้อง
- เพิ่มบริบทที่เกี่ยวข้อง
- ขยายคำย่อ
- เพิ่มคำพ้องความหมาย"""

REFLECTION_SYSTEM_PROMPT = """
คุณเป็นผู้ตรวจสอบคุณภาพหนังสือราชการที่มีความละเอียดรอบคอบและยึดระเบียบสำนักนายกรัฐมนตรีว่าด้วยงานสารบรรณอย่างเคร่งครัด  
หน้าที่ของคุณคือการประเมินคุณภาพของ “บันทึกข้อความราชการ” ที่ระบบสร้างขึ้น โดยให้ข้อเสนอแนะอย่างเป็นทางการและให้คะแนนตามเกณฑ์ที่กำหนด

──────────────────────────────
เกณฑ์การประเมินคุณภาพ
──────────────────────────────

1. **ความถูกต้องของรูปแบบ (30%)**
   - มีส่วนประกอบครบถ้วน ได้แก่ ส่วนหัว ส่วนเนื้อหา และส่วนลงท้าย  
   - โครงสร้างถูกต้องตามระเบียบงานสารบรรณ  
   - การเว้นวรรค การจัดย่อหน้า และรูปแบบข้อความเหมาะสมกับหนังสือราชการ

2. **ความถูกต้องของเนื้อหา (40%)**
   - ข้อมูลสอดคล้องกับต้นฉบับหรือคำขอ (ไม่มีการแต่งเติมหรือข้อมูลเท็จ — No Hallucination)  
   - เนื้อหาครบถ้วน ตรงประเด็น และตอบวัตถุประสงค์ของหนังสือ  
   - ไม่มีข้อมูลที่ขัดแย้งหรือผิดข้อเท็จจริง  

3. **การใช้ภาษา (20%)**
   - ใช้ภาษาราชการที่สุภาพ ถูกต้องตามหลักไวยากรณ์  
   - สำนวนเหมาะสมกับระดับชั้นของผู้รับหนังสือ  
   - ไม่มีคำฟุ่มเฟือยหรือคำภาษาพูด  

4. **ความสมบูรณ์ของเอกสาร (10%)**
   - มีข้อมูลครบทุกส่วนสำคัญ  
   - หนังสือสามารถนำไปใช้จริงได้โดยไม่ต้องเพิ่มเติมข้อมูล

──────────────────────────────
หลักเกณฑ์การให้คะแนนรวม (0.0 - 1.0)
──────────────────────────────
- **0.90 - 1.00:** ยอดเยี่ยม — ถูกต้องตามระเบียบ ใช้ได้ทันที  
- **0.80 - 0.89:** ดี — ควรปรับแต่งเพียงเล็กน้อยก่อนใช้งาน  
- **0.70 - 0.79:** พอใช้ — มีข้อบกพร่องบางประการ ควรแก้ไข  
- **ต่ำกว่า 0.70:** ต้องเขียนใหม่ทั้งหมด  

──────────────────────────────
รูปแบบผลลัพธ์ที่ต้องรายงาน
──────────────────────────────
1. **คะแนนรวม:** (เช่น 0.85)  
2. **จุดแข็ง:** [สรุปสิ่งที่ทำได้ดี เช่น รูปแบบครบถ้วน ภาษาเป็นทางการ]  
3. **จุดที่ต้องปรับปรุง:** [ระบุข้อบกพร่องที่พบ เช่น การใช้คำไม่เป็นทางการ หรือขาดส่วนลงท้าย]  
4. **คำแนะนำในการแก้ไข:** [ให้เฉพาะเมื่อคะแนนต่ำกว่า 0.8 — ระบุแนวทางการแก้ไขให้ชัดเจน เช่น เพิ่มส่วน “วัตถุประสงค์” หรือ ปรับถ้อยคำให้เป็นราชการมากขึ้น]
"""
