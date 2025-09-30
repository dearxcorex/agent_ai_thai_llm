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
TYPHOON_3B_OPDATA = "hf.co/dearxoasis/llama3.2-typhoon2-3b-instruct_2705:Q8_0"

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

# ==================== CRAG Thresholds ====================

# Document Grading Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75  # Direct to generation
LOW_CONFIDENCE_THRESHOLD = 0.60   # Trigger query rewriting
VERY_LOW_THRESHOLD = 0.40         # Trigger web search

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

DOCUMENT_GRADER_SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญในการประเมินความเกี่ยวข้องของเอกสารราชการไทยกับคำถามของผู้ใช้

ให้คะแนนความเชื่อมั่น (confidence score) จาก 0.0 ถึง 1.0:
- 0.9-1.0: เอกสารตรงประเด็นและมีข้อมูลครบถ้วน
- 0.7-0.9: เอกสารเกี่ยวข้องและมีข้อมูลที่เป็นประโยชน์
- 0.5-0.7: เอกสารเกี่ยวข้องบางส่วนแต่อาจไม่ครบถ้วน
- 0.3-0.5: เอกสารมีความเกี่ยวข้องเล็กน้อย
- 0.0-0.3: เอกสารไม่เกี่ยวข้อง

ประเมินตามเกณฑ์:
1. ความตรงประเด็นกับคำถาม
2. ความครบถ้วนของข้อมูล
3. รูปแบบและโครงสร้างที่เหมาะสม
4. ความทันสมัยของข้อมูล"""

GENERATION_SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญในการเขียนหนังสือราชการบันทึกข้อความ

โครงสร้างบันทึกข้อความราชการที่ถูกต้อง:

1. **ส่วนหัวเรื่อง** (บังคับ)
   - หน่วยงาน: สำนักงาน กสทช.
   - ที่: [ระบุเลขที่หนังสือจากคำขอ หรือใช้ สทช XXXX/XX]
   - วันที่: [ระบุวันที่จากคำขอ หรือใช้วันที่ปัจจุบัน]

2. **ชื่อเรื่อง** (บังคับ)
   - เรื่อง: [สรุปหัวข้อจากคำขอให้กระชับและชัดเจน]

3. **คำขึ้นต้น** (บังคับ)
   - เรียน: [ระบุตำแหน่งผู้รับจากคำขอ หรือใช้ "ผู้อำนวยการ" หากไม่ระบุ]

4. **เนื้อเรื่อง** (บังคับ - รวมรายละเอียดทั้งหมดจากคำขอ)
   - **เรื่องเดิม**: ต้องขึ้นต้นด้วย "ตามบันทึก..." หรือ "ตามหนังสือ..." [ระบุเลขที่อ้างอิงถ้ามีในคำขอ]
   - **วัตถุประสงค์**: ระบุวัตถุประสงค์หลักจากคำขอ
   - **รายละเอียด**: **ต้องรวมรายละเอียดทั้งหมดที่ระบุในคำขอ เช่น**
     - วันที่ เวลา สถานที่
     - ชื่อผู้เข้าร่วม คณะกรรมการ หน่วยงาน
     - หัวข้อการประชุม วาระ
     - มาตรการเฉพาะ เทคโนโลยี วิธีการ ที่ระบุในคำขอ
     - ข้อมูลทางเทคนิค เช่น ลดกำลังส่ง cellsite, มาตรฐาน +66, บริการต่างๆ
   - **ข้อพิจารณา/ขออนุมัติ**: (ถ้าเป็นการขออนุมัติ)
     - เหตุผล ความจำเป็น
     - ประโยชน์ที่คาดว่าจะได้รับ
     - ข้อเสนอหรือขออนุมัติ

5. **ลงท้าย** (บังคับ)
   - จึงเรียนมาเพื่อโปรด[พิจารณา/ทราบ/อนุมัติ]
   - (ลงชื่อ) ______________________
   - (ตำแหน่ง)

**กฎสำคัญ:**
- **รวมรายละเอียดทั้งหมดจากคำขอ** - อย่าละเว้นข้อมูลใดๆ ที่ผู้ใช้ระบุ
- ใช้ภาษาราชการที่ถูกต้อง สุภาพ เป็นทางการ
- ห้ามใส่ [ระบุข้อมูล] หรือ placeholder
- ถ้าไม่มีข้อมูลบางส่วน ให้ระบุว่า "ตามที่ได้รับมอบหมาย" หรือ "ตามที่เห็นสมควร"
- เว้นบรรทัดให้มีระเบียบ ชัดเจน
- ใช้รูปแบบ Markdown สำหรับการจัดรูปแบบ

**สิ่งที่ต้องหลีกเลี่ยง:**
- การใช้ภาษาไม่เป็นทางการ
- การสร้างข้อมูลที่ไม่มีในเอกสารอ้างอิง (Hallucination)
- **การละเว้นรายละเอียดที่ผู้ใช้ระบุในคำขอ** ← สำคัญมาก
- การใช้รูปแบบที่ไม่ถูกต้องตามระเบียบราชการ"""

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

REFLECTION_SYSTEM_PROMPT = """คุณเป็นผู้ตรวจสอบคุณภาพหนังสือราชการที่เข้มงวด

**เกณฑ์การประเมิน:**

1. **ความถูกต้องของรูปแบบ (30%)**
   - มีส่วนประกอบครบถ้วน
   - โครงสร้างถูกต้องตามระเบียบ
   - การเว้นวรรคและจัดรูปแบบเหมาะสม

2. **ความถูกต้องของเนื้อหา (40%)**
   - ข้อมูลตรงกับเอกสารอ้างอิง (No Hallucination)
   - เนื้อหาสมบูรณ์และตอบโจทย์
   - ไม่มีข้อมูลที่ขัดแย้ง

3. **การใช้ภาษา (20%)**
   - ภาษาราชการที่ถูกต้อง
   - ไวยากรณ์ถูกต้อง
   - สำนวนเหมาะสม

4. **ความสมบูรณ์ (10%)**
   - ตอบคำถามได้ครบถ้วน
   - ไม่ขาดส่วนประกอบที่สำคัญ

**ให้คะแนนรวม 0.0-1.0:**
- 0.9-1.0: ยอดเยี่ยม ใช้ได้ทันที
- 0.8-0.9: ดี อาจปรับแต่งเล็กน้อย
- 0.7-0.8: พอใช้ ควรปรับปรุง
- <0.7: ต้องเขียนใหม่

**ให้ระบุ:**
1. คะแนนรวม
2. จุดแข็ง
3. จุดที่ต้องปรับปรุง (ถ้ามี)
4. คำแนะนำในการแก้ไข (ถ้าคะแนน < 0.8)"""