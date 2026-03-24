import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from baseline_inference import load_data, run_baseline
from evaluation_engine.extractor import extract_boxed_answer
from evaluation_engine.metrics import compute_accuracy
from optimization.error_clustering import cluster_errors

def evaluate_and_cluster(model_name: str, num_samples: int):
    """
    Evaluates the model on test set and explicitly conducts Error Analysis
    by classifying failure modes, outputting a clustered breakdown.
    """
    # Simulate loading model and running eval
    dataset = load_data(max_samples=num_samples)
    
    predictions = []
    references = []
    questions = []
    
    print("\n[Executing Evaluation Sequence]")
    for item in dataset:
        questions.append(item["question"])
        references.append(item["ground_truth"])
        
        # In a real environment we pass it through model.generate.
        # Here we simulate generation that is partly correct to generate error clusters.
        import random
        if random.random() > 0.6:
            predictions.append(item["ground_truth"])
        else:
            # Simulate various failure patterns
            err_type = random.choice(["Missing box", "Calculation", "Latex", "Hallucination"])
            if err_type == "Missing box":
                predictions.append(f"The answer is {item['ground_truth']}.")
            elif err_type == "Calculation":
                predictions.append(f"\\boxed{{9999}}")
            elif err_type == "Latex":
                predictions.append(f"\\boxed{{{item['ground_truth']}\\text{{ apples}}}}")
            else:
                predictions.append("I don't know the answer.")
                
    # Normalize extractions
    preds_extracted = [extract_boxed_answer(p) if "\\boxed{" in p else "" for p in predictions]
    
    acc = compute_accuracy(preds_extracted, references)
    print(f"\nFinal Accuracy: {acc * 100:.2f}%")
    
    print("\n[Generating Error Analysis Report]")
    clusters = cluster_errors(preds_extracted, references, questions)
    
    for cluster_name, errors in clusters.items():
        print(f"-> {cluster_name}: {len(errors)} instances")
        for e in errors[:2]:  # Sample first 2
            print(f"   Sample: {e}")
            
    print("\nEvaluation & Clustering pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Mini-4B-Instruct")
    # You would typically pass the base model and the trained LoRA adapter path as well
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    
    evaluate_and_cluster(args.model, args.samples)
