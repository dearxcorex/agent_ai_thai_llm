"""
RAGAS Evaluation Framework Integration
Metrics for measuring RAG system accuracy and quality

Metrics:
- Faithfulness: How factually accurate is the generation?
- Answer Relevancy: How relevant is the answer to the question?
- Context Precision: How precise are the retrieved documents?
- Context Recall: How complete are the retrieved documents?
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import pandas as pd
from typing import List, Dict
import json
from datetime import datetime
from pathlib import Path
import config


class RAGEvaluator:
    """
    Evaluator for RAG system using RAGAS framework
    """

    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

    def prepare_eval_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dataset:
        """
        Prepare dataset for RAGAS evaluation

        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of context documents (list of lists)
            ground_truths: List of expected answers (optional)

        Returns:
            Dataset object for RAGAS evaluation
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths

        return Dataset.from_dict(data)

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict:
        """
        Evaluate RAG system using RAGAS metrics

        Returns:
            Dictionary with evaluation scores
        """
        # Prepare dataset
        dataset = self.prepare_eval_dataset(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )

        return result

    def save_results(self, results: Dict, output_dir: str = None):
        """
        Save evaluation results to file

        Args:
            results: RAGAS evaluation results
            output_dir: Output directory (defaults to config.EVALUATION_OUTPUT_DIR)
        """
        if output_dir is None:
            output_dir = config.EVALUATION_OUTPUT_DIR

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = Path(output_dir) / f"ragas_eval_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save as CSV
        if hasattr(results, 'to_pandas'):
            csv_path = Path(output_dir) / f"ragas_eval_{timestamp}.csv"
            results.to_pandas().to_csv(csv_path, index=False, encoding='utf-8')

        print(f"✓ Results saved to {output_dir}")
        print(f"  - JSON: {json_path}")
        if hasattr(results, 'to_pandas'):
            print(f"  - CSV: {csv_path}")


class BenchmarkRunner:
    """
    Run benchmarks to compare system performance
    """

    def __init__(self, rag_app):
        """
        Args:
            rag_app: Compiled LangGraph application
        """
        self.rag_app = rag_app
        self.evaluator = RAGEvaluator()

    def run_benchmark(
        self,
        test_cases: List[Dict],
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Run benchmark on test cases

        Args:
            test_cases: List of dicts with 'question' and optionally 'expected_answer'
            save_results: Whether to save results to file

        Returns:
            DataFrame with benchmark results
        """
        print(f"\nRunning benchmark on {len(test_cases)} test cases...")

        questions = []
        answers = []
        contexts = []
        ground_truths = []
        processing_times = []
        quality_scores = []

        for i, test_case in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] Processing: {test_case['question'][:60]}...")

            # Prepare input state
            input_state = {
                "question": test_case["question"],
                "reflection_iteration": 0,
                "web_search_needed": False,
                "query_rewrite_needed": False,
                "regeneration_needed": False,
                "warnings": []
            }

            # Run RAG system
            import time
            start_time = time.time()

            final_state = None
            for output in self.rag_app.stream(input_state):
                for node_name, node_output in output.items():
                    final_state = node_output

            processing_time = time.time() - start_time

            # Extract results
            question = test_case["question"]
            answer = final_state.get("generation", "")
            context_docs = final_state.get("filtered_docs", [])
            context = [doc.page_content for doc in context_docs]
            quality_score = final_state.get("quality_score", 0.0)

            questions.append(question)
            answers.append(answer)
            contexts.append(context)
            processing_times.append(processing_time)
            quality_scores.append(quality_score)

            if "expected_answer" in test_case:
                ground_truths.append(test_case["expected_answer"])

            print(f"  ✓ Generated ({len(answer)} chars, {processing_time:.2f}s, quality={quality_score:.2f})")

        # Run RAGAS evaluation
        print("\nRunning RAGAS evaluation...")

        eval_results = self.evaluator.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if ground_truths else None
        )

        # Create results dataframe
        results_df = eval_results.to_pandas() if hasattr(eval_results, 'to_pandas') else pd.DataFrame(eval_results)

        # Add additional metrics
        results_df['processing_time'] = processing_times
        results_df['quality_score'] = quality_scores
        results_df['answer_length'] = [len(a) for a in answers]

        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)

        print(f"\nRAGAS Metrics:")
        print(f"  Faithfulness: {results_df['faithfulness'].mean():.3f} ± {results_df['faithfulness'].std():.3f}")
        print(f"  Answer Relevancy: {results_df['answer_relevancy'].mean():.3f} ± {results_df['answer_relevancy'].std():.3f}")
        print(f"  Context Precision: {results_df['context_precision'].mean():.3f} ± {results_df['context_precision'].std():.3f}")
        print(f"  Context Recall: {results_df['context_recall'].mean():.3f} ± {results_df['context_recall'].std():.3f}")

        print(f"\nPerformance Metrics:")
        print(f"  Avg Processing Time: {results_df['processing_time'].mean():.2f}s")
        print(f"  Avg Quality Score: {results_df['quality_score'].mean():.3f}")
        print(f"  Avg Answer Length: {results_df['answer_length'].mean():.0f} chars")

        # Save results
        if save_results and config.SAVE_EVALUATION_RESULTS:
            self.evaluator.save_results(eval_results)

            # Also save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_path = Path(config.EVALUATION_OUTPUT_DIR) / f"benchmark_detail_{timestamp}.csv"
            results_df.to_csv(detail_path, index=False, encoding='utf-8')
            print(f"\n✓ Detailed results saved to {detail_path}")

        return results_df


# ==================== Example Test Cases ====================

THAI_GOV_DOC_TEST_CASES = [
    {
        "question": "เขียนหนังสือบันทึกข้อความ ขออนุมัติเดินทางตรวจตามแผน",
        "expected_answer": None  # Optional ground truth
    },
    {
        "question": "สร้างหนังสือราชการเชิญประชุมกับคณะกรรมการ ITU โดยมีรายละเอียด การป้องกัน แก๊ง call center",
        "expected_answer": None
    },
    {
        "question": "สร้างหนังสือราชการบันทึกข้อความขออนุมัติเดินทางตรวจสอบตามแผนประจำปี 2568 โดยมีข้อมูล เรื่องตรวจสอบตามแผน อ้างอิง สทช2203/144 เดินทางจังหวัดบุรีรัมย์",
        "expected_answer": None
    },
    {
        "question": "เขียนบันทึกข้อความขออนุมัติจัดซื้อจัดจ้าง",
        "expected_answer": None
    },
    {
        "question": "สร้างหนังสือราชการรายงานผลการปฏิบัติงาน",
        "expected_answer": None
    }
]


# ==================== Main Execution ====================

if __name__ == "__main__":
    print("="*80)
    print("RAGAS EVALUATION FRAMEWORK")
    print("="*80)

    # This is a standalone example
    # In practice, you would import the compiled RAG app from corrective_rag_web_search.py

    print("\nTo run evaluation:")
    print("1. Import your compiled LangGraph app")
    print("2. Create BenchmarkRunner instance")
    print("3. Run benchmark with test cases")

    print("\nExample usage:")
    print("""
from evaluation import BenchmarkRunner, THAI_GOV_DOC_TEST_CASES
from corrective_rag_web_search import app

# Create benchmark runner
runner = BenchmarkRunner(app)

# Run benchmark
results = runner.run_benchmark(
    test_cases=THAI_GOV_DOC_TEST_CASES,
    save_results=True
)

# View results
print(results)
""")

    # Example of simple evaluation
    print("\n" + "="*80)
    print("SIMPLE RAGAS EVALUATION EXAMPLE")
    print("="*80)

    evaluator = RAGEvaluator()

    # Example data
    sample_data = {
        "questions": ["What is the capital of France?"],
        "answers": ["The capital of France is Paris."],
        "contexts": [["Paris is the capital and most populous city of France."]],
    }

    print("\nSample evaluation (requires OpenAI-compatible API for RAGAS)...")
    print("Note: RAGAS uses LLM for evaluation metrics (DeepSeek supported)")

    try:
        result = evaluator.evaluate(
            questions=sample_data["questions"],
            answers=sample_data["answers"],
            contexts=sample_data["contexts"]
        )
        print("\nResults:")
        print(result)
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print("Make sure you have set DEEPSEEK_API_KEY for RAGAS evaluation")