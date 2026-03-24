import json
import argparse
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data_engine.generator import generate_synthetic_math_problems
from data_engine.filter import filter_high_quality
from self_refinement.loop import run_refinement_logic
from optimization.novel_techniques import dynamic_quality_scoring

def build_dataset(num_samples: int, output_file: str, use_mock: bool = True):
    print(f"Generating {num_samples} seed synthetic problems (Arithmetic/Logic/Symbolic)...")
    raw_problems = generate_synthetic_math_problems(num_samples)
    
    # 2. Heuristic Initial Filter (ensures ground truth is structured)
    filtered_problems = filter_high_quality(raw_problems)
    print(f"Initial strict-filtering retained {len(filtered_problems)} base seeds.")
    
    if use_mock:
        def model_generate(prompt):
            if "Critique" in prompt:
                # Simulated Critic: Identifies logic flaw half the time
                import random
                if random.random() > 0.5:
                    return "CORRECT"
                return "You forgot an intermediate step in the calculation."
            # Simulated Generator extracting the ground truth correctly
            import re
            match = re.search(r'Problem:\n(.*)\n', prompt)
            
            # Very basic mock response 
            return "Step 1: Understand problem. Step 2: Use established templates. Thus the answer is \\boxed{MOCK_ANSWER}."
            
            return "Step 1: Parse the parameters. We calculate the result based on the operations. Thus the answer is \\boxed{42}."
    else:
        # Load real model code path
        import torch
        from transformers import pipeline
        print("Loading Nemotron Model for dataset generation...")
        try:
            pipe = pipeline(
                "text-generation", 
                model="nvidia/Nemotron-Mini-4B-Instruct", 
                device_map="auto",
                model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True}
            )
            def model_generate(prompt):
                out = pipe(prompt, max_new_tokens=512, temperature=0.0, do_sample=False)
                return out[0]["generated_text"][len(prompt):]
        except Exception as e:
            print(f"Real model load failed ({e}), aborting generation run.")
            return

    print(f"\nRunning Sub-Agent Self-Refinement loop on seeds...")
    final_dataset = []
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    rejected_count = 0
    
    for item in tqdm(filtered_problems):
        # The engine drives the model to solve, critique, and improve its answer
        res = run_refinement_logic(item["question"], model_generate, max_iters=2)
        
        trace = res["final_solution"]
        
        # Validate that the model successfully boxed an answer
        if "\\boxed{" not in trace:
            rejected_count += 1
            continue
            
        # Optimization: Dynamic Quality Scoring system 
        score = dynamic_quality_scoring({"reasoning_trace": trace})
        
        # Keep only top quality / fully structured samples
        if score >= 0.2:
            final_dataset.append({
                "instruction": "Solve step by step and give final answer in \\boxed{}",
                "input": item["question"],
                "output": trace,
                "score": score
            })
        else:
            rejected_count += 1
            
    print(f"\nFinal high-quality dataset size: {len(final_dataset)} (Rejected: {rejected_count})")
    
    print(f"Formatting to SFT format and saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for d in final_dataset:
            f.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Number of seed problems to generate")
    parser.add_argument("--out", type=str, default="data/processed/refined_dataset.jsonl", help="Output SFT dataset path")
    parser.add_argument("--mock", action="store_true", help="Run with mock responses for architecture validation.")
    args = parser.parse_args()
    build_dataset(args.samples, args.out, args.mock)
