import json
import re

def run_distillation():
    source_path = "data/processed/augmented_train.json"
    out_path = "data/processed/distilled_dataset.json"
    
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    distilled = []
    
    # Target exact mapping efficiency without verbose language models mapping overhead
    strict_instruction = "Analyze transformation rules. Output ONLY Step 1, Step 2, and Final target in \\boxed{} format explicitly."
    
    for item in data:
        trace = item["output"]
        
        ans_match = re.search(r'\\boxed\{([^}]*)\}', trace)
        if not ans_match: continue
            
        ans = ans_match.group(1)
        
        # Distill trace payload to bare mechanical essence prioritizing computation mapping
        distilled_trace = f"Step 1: Identify Transformation Rule Mapping.\nStep 2: Apply Target Transformation unconditionally.\nFinal Answer: \\boxed{{{ans}}}"
        
        distilled.append({
            "instruction": strict_instruction,
            "input": item["input"],
            "output": distilled_trace
        })
        
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(distilled, f, indent=2)
        
    print(f"Reasoning Trace Distillation complete. Stripped {len(data)} verbose sequences into {len(distilled)} strictly targeted concise logic mappings.")
    print(f"Saved optimized SFT payload to {out_path}.")

if __name__ == "__main__":
    run_distillation()
