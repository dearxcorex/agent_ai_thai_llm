# Migration from OpenAI to DeepSeek

This document outlines the changes made to replace OpenAI with DeepSeek in the Advanced Agentic RAG system.

## Summary of Changes

The system has been updated to use **DeepSeek** instead of OpenAI for document grading, query optimization, and self-reflection agents. DeepSeek offers cost-effective LLM services with OpenAI-compatible API endpoints.

## Changes Made

### 1. Configuration ([config.py](config.py))

**Before:**
```python
# OpenAI Models
GRADING_MODEL = "gpt-4o-mini"
VALIDATION_MODEL = "gpt-4o-mini"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validation
required_keys = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    ...
}
```

**After:**
```python
# DeepSeek Models
GRADING_MODEL = "deepseek-chat"
VALIDATION_MODEL = "deepseek-chat"

# API Keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Validation
required_keys = {
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    ...
}
```

### 2. Main RAG System ([corrective_rag_web_search.py](corrective_rag_web_search.py))

**Before:**
```python
from langchain_openai import ChatOpenAI

# Grading and Optimization LLM (OpenAI)
llm_small = ChatOpenAI(
    model=config.GRADING_MODEL,
    temperature=0
)
```

**After:**
```python
from langchain_openai import ChatOpenAI  # Using OpenAI-compatible API for DeepSeek

# Grading and Optimization LLM (DeepSeek via OpenAI-compatible API)
llm_small = ChatOpenAI(
    model=config.GRADING_MODEL,
    base_url="https://api.deepseek.com",
    api_key=config.DEEPSEEK_API_KEY,
    temperature=0
)
```

### 3. Requirements ([requirements.txt](requirements.txt))

**Updated comment:**
```python
# LLM Providers
langchain-openai>=0.2.0  # For DeepSeek (OpenAI-compatible API)
```

### 4. Documentation ([README_ADVANCED_RAG.md](README_ADVANCED_RAG.md))

**Updated sections:**
- Prerequisites: Changed "OpenAI API key" → "DeepSeek API key"
- Setup instructions: Updated `.env` example
- Configuration: Updated model settings
- Troubleshooting: Updated RAGAS evaluation notes

### 5. Evaluation ([evaluation.py](evaluation.py))

**Updated comments:**
```python
print("Note: RAGAS uses LLM for evaluation metrics (DeepSeek supported)")
print("Make sure you have set DEEPSEEK_API_KEY for RAGAS evaluation")
```

## Environment Setup

Update your `.env` file:

**Before:**
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxx
TAVILY_API_KEY=xxxxxxxxxxxxx
```

**After:**
```bash
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxx
TAVILY_API_KEY=xxxxxxxxxxxxx
```

## Getting DeepSeek API Key

1. Visit [DeepSeek Platform](https://platform.deepseek.com)
2. Sign up / Log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add to your `.env` file

## Why DeepSeek?

### Advantages:

1. **Cost-Effective**: Significantly cheaper than OpenAI GPT-4o-mini
2. **OpenAI-Compatible**: Drop-in replacement using OpenAI SDK
3. **Good Performance**: Competitive accuracy for Thai language tasks
4. **No Lock-in**: Easy to switch back to OpenAI if needed

### Pricing Comparison (approximate):

| Provider | Model | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|-------|----------------------|----------------------|
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| DeepSeek | deepseek-chat | $0.14 | $0.28 |

*DeepSeek offers ~50% cost savings on output tokens*

## Testing

After migration, verify the system works:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Verify configuration
python -c "from config import validate_config; validate_config()"

# 3. Run test
uv run corrective_rag_web_search.py
```

Expected output should show:
```
✓ Configuration validated successfully
Loading documents...
Loaded X documents
Setting up hybrid retrieval system...
✓ Hybrid retrieval system ready
Building LangGraph workflow...
✓ LangGraph workflow compiled successfully
```

## Rollback (if needed)

To switch back to OpenAI:

1. Update [config.py](config.py:18-20):
   ```python
   # OpenAI Models
   GRADING_MODEL = "gpt-4o-mini"
   VALIDATION_MODEL = "gpt-4o-mini"
   ```

2. Update [config.py](config.py:109-122):
   ```python
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

   required_keys = {
       "OPENAI_API_KEY": OPENAI_API_KEY,
       ...
   }
   ```

3. Update [corrective_rag_web_search.py](corrective_rag_web_search.py:226-231):
   ```python
   llm_small = ChatOpenAI(
       model=config.GRADING_MODEL,
       temperature=0
   )
   ```

4. Update `.env`:
   ```bash
   OPENAI_API_KEY=sk-xxxxxxxxxxxxx
   ```

## Performance Notes

- **DeepSeek** performs well on Thai language tasks
- If you encounter quality issues, you can:
  - Adjust prompts in [config.py](config.py:136-238)
  - Increase `REFLECTION_THRESHOLD` for stricter validation
  - Use hybrid approach (DeepSeek for grading, OpenAI for validation)

## Support

For DeepSeek-specific issues:
- Documentation: https://platform.deepseek.com/docs
- API Status: https://status.deepseek.com

For system issues:
- Check logs for detailed error messages
- Verify API keys are correctly set
- Ensure sufficient API credits

---

**Migration Date**: 2025-09-30
**Version**: 2.0.1 (DeepSeek Integration)