import sys, os, json
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from training_engine.train_lora import train_lora_model

def run():
    print("Loading heavily augmented JSON training pipeline dataset...")
    path = "data/processed/augmented_train.json"
    if not os.path.exists(path):
         print("Missing augmented dataset target")
         return
         
    with open(path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
        
    mapped_data = [{"question": d["input"], "reasoning_trace": d["output"]} for d in raw_data]
        
    print(f"Deploying QLoRA Rank constraint retraining on {len(mapped_data)} structurally enriched targets...")
    train_lora_model("nvidia/Nemotron-Mini-4B-Instruct", mapped_data, "models/transformation_lora_augmented")

if __name__ == "__main__":
    run()
