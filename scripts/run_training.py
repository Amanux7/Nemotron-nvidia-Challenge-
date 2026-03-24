import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from training_engine.train_lora import train_lora_model

def load_jsonl(filepath: str) -> list:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_training_pipeline(dataset_path: str, model_id: str, output_dir: str):
    if not os.path.exists(dataset_path):
        print(f"Dataset block missing: {dataset_path}")
        return
        
    print(f"Loading generated/refined dataset from {dataset_path}...")
    dataset = load_jsonl(dataset_path)
    
    # Map back to expected internal structure for train_lora_model 
    # train_lora_model expects {"question": "...", "reasoning_trace": "..."}
    internal_data = []
    for d in dataset:
        internal_data.append({
            "question": d["input"],
            "reasoning_trace": d["output"]
        })
        
    print(f"Training LoRA targeting '{model_id}' on {len(internal_data)} structured examples...")
    train_lora_model(model_id, internal_data, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/refined_dataset.jsonl", help="Path to high quality SFT jsonl")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Mini-4B-Instruct", help="Base model to apply LoRA constraint to")
    parser.add_argument("--out", type=str, default="models/trained_lora", help="Output directory")
    args = parser.parse_args()
    
    run_training_pipeline(args.data, args.model, args.out)
