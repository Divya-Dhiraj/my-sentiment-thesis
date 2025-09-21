# test/evaluate_agent.py
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    answer_correctness
)
from evaluation_dataset import EVAL_DATASET

AGENT_API_URL = "http://agent_app:8001/ask_agent"

def run_evaluation():
    """
    Runs the agent on a set of test questions and evaluates the responses using Ragas.
    """
    print("--- 🚀 Starting Agent Evaluation ---")
    
    results = []
    print(f"--- 📞 Calling Agent for {len(EVAL_DATASET)} questions... ---")
    for i, data_row in enumerate(EVAL_DATASET):
        question = data_row["question"]
        print(f"--- Running Q{i+1}: {question} ---")
        try:
            response = requests.post(AGENT_API_URL, json={"query": question}, timeout=300)
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("analysis_summary", "")
            
            # --- FINAL FIX: Ensure contexts are always a clean list of non-empty strings ---
            contexts = []
            for step in data.get("steps", []):
                # Ensure the output is a non-empty string before adding
                if isinstance(step.get("output"), str) and step["output"]:
                    contexts.append(step["output"])
            
            if not contexts:
                contexts = ["No context was retrieved."] # Provide a default non-empty context for Ragas

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": data_row["ground_truth"]
            })
            print(f"--- ✅ Agent answered successfully. ---")

        except Exception as e:
            print(f"--- ❌ ERROR: Failed to get answer for question: {question} ---")
            print(f"Error: {e}")
            results.append({
                "question": question,
                "answer": f"AGENT FAILED: {e}",
                "contexts": ["Error retrieving context."],
                "ground_truth": data_row["ground_truth"]
            })
            
    # Filter out any failed runs before sending to Ragas
    successful_results = [res for res in results if "AGENT FAILED" not in res["answer"]]
    if not successful_results:
        print("\n--- ⚠️ No successful agent runs to evaluate. ---")
        return

    print("\n--- 📊 Preparing data for Ragas evaluation... ---")
    results_dataset = Dataset.from_pandas(pd.DataFrame(successful_results))
    print("Dataset prepared for Ragas:")
    print(results_dataset)

    print("\n--- 🧪 Running Ragas evaluation... ---")
    
    result = evaluate(
        dataset=results_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, answer_correctness],
    )
    
    print("\n--- 🏁 Evaluation Complete ---")
    print("--- Here are the scores for your agent (1.0 is a perfect score) ---")
    print(result)
    
    return result

if __name__ == "__main__":
    run_evaluation()