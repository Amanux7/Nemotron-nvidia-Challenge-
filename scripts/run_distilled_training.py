import sys, os, json
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from training_engine.train_lora import train_lora_model

def run():
    print("Loading DISTILLED precision JSON SFT dataset sequences...")
    path = "data/processed/distilled_dataset.json"
         
    with open(path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
        
    mapped_data = [{"question": d["input"], "reasoning_trace": d["output"]} for d in raw_data]
        
    print(f"Deploying QLoRA Rank constraint isolated retraining against {len(mapped_data)} optimal precision targets (Epoch 3)...")
    train_lora_model("nvidia/Nemotron-Mini-4B-Instruct", mapped_data, "models/transformation_lora_distilled")

if __name__ == "__main__":
    run()
