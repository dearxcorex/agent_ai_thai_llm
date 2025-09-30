# Thai Government Document RAG System

Advanced Retrieval Augmented Generation system for generating Thai government official documents (หนังสือราชการ).

## 📁 Project Structure

```
.
├── config/
│   └── config.py              # Centralized configuration (models, thresholds, prompts)
│
├── data_markdown/             # Source documents for RAG retrieval
│   └── *.md                   # Thai government document examples
│
├── docs/                      # Documentation
│   ├── README_ADVANCED_RAG.md    # System architecture & usage
│   ├── MODEL_COMPARISON.md       # DeepSeek vs Ollama Typhoon comparison
│   ├── MIGRATION_TO_DEEPSEEK.md  # Migration guide
│   └── CLAUDE.md                 # Project instructions for Claude Code
│
├── output/
│   ├── results/               # Generated documents
│   └── logs/                  # Execution logs
│
├── sample/                    # Sample/testing code
│   ├── agentic_rag.py
│   └── corrective_RAG.ipynb
│
├── corrective_rag_web_search.py  # Main RAG system
├── evaluation.py              # RAGAS evaluation framework
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate UV virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create/update `.env` file:

```bash
# Required
DEEPSEEK_API_KEY=your_deepseek_key
COHERE_API_KEY=your_cohere_key
TAVILY_API_KEY=your_tavily_key

# Optional (for LangSmith tracing)
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=agentic_rag
```

### 3. Run the RAG System

```bash
uv run corrective_rag_web_search.py
```

You'll be prompted to select a generation model:
- **Option 1: DeepSeek** (Recommended) - Quality score 0.75 ✅
- **Option 2: Ollama Typhoon 3B** - Quality score 0.65 ⚠️

See [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) for detailed comparison.

## 📊 System Features

### Advanced RAG Components

1. **Corrective RAG (CRAG)** - Confidence-scored document grading
2. **Hybrid Retrieval** - BM25 + Dense vectors + Cohere Reranking
3. **Query Optimization** - Automatic query rewriting
4. **Self-Reflection** - Quality validation before output
5. **Web Search Fallback** - Tavily integration
6. **RAGAS Evaluation** - Comprehensive accuracy metrics

### Supported Document Types

- บันทึกข้อความ (Internal memo)
- หนังสือเชิญประชุม (Meeting invitation)
- หนังสือขออนุมัติ (Approval request)
- หนังสือรายงาน (Report)

## 📖 Documentation

- **[Advanced RAG System](docs/README_ADVANCED_RAG.md)** - Complete system documentation
- **[Model Comparison](docs/MODEL_COMPARISON.md)** - DeepSeek vs Ollama analysis
- **[Migration Guide](docs/MIGRATION_TO_DEEPSEEK.md)** - OpenAI to DeepSeek migration

## 🛠️ Configuration

Main configuration in [config/config.py](config/config.py):

```python
# Model Selection
GENERATION_MODEL = "deepseek-chat"      # Or use Typhoon 3B via Ollama
GRADING_MODEL = "deepseek-chat"
VALIDATION_MODEL = "deepseek-chat"

# CRAG Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75        # Generate
LOW_CONFIDENCE_THRESHOLD = 0.60         # Query rewrite
VERY_LOW_THRESHOLD = 0.40               # Web search

# Self-Reflection
REFLECTION_THRESHOLD = 0.7              # Minimum quality score
MAX_REFLECTION_ITERATIONS = 3
```

## 📈 Performance

| Metric | Baseline | Enhanced |
|--------|----------|----------|
| Retrieval Precision | ~60% | 85%+ |
| Answer Groundedness | 0.70 | 0.90+ |
| Factual Accuracy | ~70% | 90%+ |
| Format Compliance | ~75% | 95%+ |

## 🎯 Usage Examples

### Generate a Meeting Invitation

```python
input_state = {
    "question": "เขียนหนังสือเชิญประชุมคณะกรรมการ ITU เรื่องมาตรการป้องกันแก๊งคอลเซ็นเตอร์"
}

for output in app.stream(input_state, config):
    print(output)
```

### Evaluate with RAGAS

```python
from evaluation import RAGEvaluator, BenchmarkRunner

evaluator = RAGEvaluator()
runner = BenchmarkRunner(evaluator, rag_system=app)

results = runner.run_benchmark(test_cases)
runner.save_results(results, "evaluation_results.json")
```

## 🐛 Troubleshooting

### Import Error after folder restructure:
```bash
# Update Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Config import issues:
The system expects `config/config.py`. If you see import errors, check that:
1. The `config/` folder exists
2. config/config.py contains all required variables
3. Import statement is: `from config import config as app_config`

## 📝 Development

### Adding New Document Types

1. Add examples to `data_markdown/`
2. Update `GENERATION_SYSTEM_PROMPT` in `config/config.py`
3. Add test cases to `evaluation.py`

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **LangChain** - RAG framework
- **DeepSeek** - LLM API
- **Cohere** - Embeddings and reranking
- **Tavily** - Web search
- **Ollama** - Local model inference