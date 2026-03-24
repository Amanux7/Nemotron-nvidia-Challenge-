import sys, os, json
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from training_engine.train_lora import train_lora_model

def run():
    print("Loading heavily filtered training JSON dataset...")
    with open("data/processed/train_dataset.json", "r", encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # Map back to structural expectations
    mapped_data = []
    for d in raw_data:
        mapped_data.append({"question": d["input"], "reasoning_trace": d["output"]})
        
    print(f"Initiating strict QLoRA training sequence targeting constraint ranks...")
    print(f"Data samples passed to SFTTrainer: {len(mapped_data)}")
    
    train_lora_model("nvidia/Nemotron-Mini-4B-Instruct", mapped_data, "models/transformation_lora")

if __name__ == "__main__":
    run()
