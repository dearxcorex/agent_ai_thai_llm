# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval Augmented Generation) system that generates Thai government official documents (บันทึกข้อความราชการ) using LangChain, LangGraph, and various LLM providers. The system implements a Corrective RAG pattern with web search fallback.

## Environment Setup

All commands must be run in a UV virtual environment:
```bash
# Activate venv
source .venv/bin/activate
```

## Running the Application

Execute the main RAG pipeline:
```bash
uv run corrective_rag_web_search.py
```

## Architecture

### Main Pipeline (corrective_rag_web_search.py)

The system implements a LangGraph-based workflow with the following nodes:

1. **retrieve** - Retrieves documents from FAISS vector store using Cohere multilingual embeddings
2. **grade_documents** - Grades document relevance using GPT-4o-mini with structured output
3. **web_search_node** - Falls back to Tavily web search if documents aren't relevant
4. **generate** - Generates Thai government document using Ollama (Typhoon model)

### State Flow

```
START → retrieve → grade_documents → [conditional]
                                    ↓
                    web_search_node OR generate
                                    ↓
                                   END
```

### Key Components

- **Vector Store**: FAISS with Cohere embeddings (embed-multilingual-v3.0)
- **Document Loading**: Markdown files from `./data_markdown` directory
- **Chunking**: RecursiveCharacterTextSplitter (chunk_size=1024, overlap=250)
- **LLM Models**:
  - Document grading: GPT-4o-mini (via OpenAI)
  - Generation: Ollama with Typhoon 3B model (quantized Q8_0)
- **Web Search**: TavilySearchResults (k=3)

### Document Grading Logic

The grader evaluates document relevance with binary scores ("ใช่"/"ไม่ใช่"). If all documents are irrelevant, the system triggers web search. The grader uses Thai language prompts and is not overly strict to avoid filtering useful context.

### Generation Format

The system generates Thai government documents with this structure:
1. ส่วนหัวเรื่อง (header: agency, document number, date)
2. ชื่อเรื่อง (subject)
3. คำขึ้นต้น (salutation: เรียน)
4. เนื้อเรื่อง (body: เรื่องเดิม, ข้อพิจารณา)
5. ลงชื่อและตำแหน่ง (signature and position)

## Data Directory Structure

- `data_markdown/` - Source documents for RAG (Thai government document examples)
- `result/` - Output directory for generated documents
- `sample/` - Contains sample/testing code (agentic_rag.py, corrective_RAG.ipynb)
- `chroma_db/` - Previously used Chroma vector store (now using FAISS in memory)

## Environment Variables

Required API keys in `.env`:
- `OPENAI_API_KEY` - For GPT-4o-mini document grading
- `COHERE_API_KEY` - For multilingual embeddings
- `TAVILY_API_KEY` - For web search fallback
- `LANGCHAIN_API_KEY` - For LangSmith tracing
- `LANGCHAIN_PROJECT` - Set to "agentic_rag"

Optional:
- `TYPHOON_API_KEY` - If using Typhoon via OpenRouter instead of Ollama
- `LANGGRAPH_API_KEY` - For LangGraph Cloud features

## Model Configuration

The system uses local Ollama models for generation. Current model:
- `hf.co/dearxoasis/llama3.2-typhoon2-3b-instruct_2705:Q8_0` (TYPHOON_3B_OPDATA)

Alternative model available:
- `hf.co/dearxoasis/llama3.2-typhoon2-offical_doc-3b:Q8_0` (TYPHOON_3B)

To switch models, change the model parameter in the ChatOllama initialization (line 32-38).

## Testing Inputs

The code includes three sample queries at the end:
- `input_state_1`: Request for travel approval document
- `input_state_2`: Request for meeting invitation regarding ITU committee and call center prevention
- `input_state_4`: Travel approval with specific dates and reference numbers

To test different queries, modify the input at line 307.

## LangSmith Tracing

LangSmith tracing is enabled by default. View traces at https://smith.langchain.com with project "agentic_rag".

## Important Notes

- The system outputs to stdout - the final generation is printed at line 311
- Document relevance grading uses Thai language prompts exclusively
- Web search is triggered only if no relevant documents are found
- The generation model (Ollama) runs with temperature=0 for deterministic outputs
- Markdown output format is enforced for structured government document formatting