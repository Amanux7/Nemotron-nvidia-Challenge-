import os
import sys
import argparse
from tqdm import tqdm
import torch

# Ensure src is in path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_engine.templates import BASELINE_COT_PROMPT
from evaluation_engine.extractor import extract_boxed_answer
from evaluation_engine.metrics import compute_accuracy

def load_data(split="test", max_samples=10):
    """
    Loads baseline evaluation data.
    Uses GSM8K as a proxy for logic/math reasoning datasets.
    """
    try:
        from datasets import load_dataset
        print("Loading GSM8K dataset...")
        ds = load_dataset("gsm8k", "main", split=split)
    except Exception as e:
        print(f"Failed to load datasets via HuggingFace: {e}")
        print("Using dummy fallback dataset.")
        ds = [{"question": "What is 2+2?", "answer": "4"}, 
              {"question": "If John has 5 apples and eats 2, how many remain?", "answer": "3"}]
        
    samples = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
            
        ans = item.get("answer", "")
        # GSM8K uses #### to delimit the final answer
        if "#### " in ans:
            ans = ans.split("#### ")[-1].strip()
            
        samples.append({
            "question": item["question"],
            "ground_truth": ans
        })
    return samples

def run_baseline(model_name: str, num_samples: int):
    print(f"Initializing run for model: {model_name}")
    
    try:
        if model_name == "mock": raise Exception("mock")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model (4-bit quantization if bitsandbytes is available)...")
        # For Nemotron, bitsandbytes quantization is crucial for local testing
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_4bit=True
            )
        except ValueError:
            print("Running without 4-bit load (fallback)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype=torch.float16
            )
    except Exception as e:
        print(f"Failed to load model locally: {e}")
        print("Executing in mock-mode for testing the pipeline flow...")
        model, tokenizer = None, None

    dataset = load_data(max_samples=num_samples)
    
    predictions = []
    references = []
    
    print("\nStarting evaluation loop...")
    for item in tqdm(dataset):
        prompt = BASELINE_COT_PROMPT.format(problem=item["question"])
        
        response = ""
        if model is not None and tokenizer is not None:
            # Tokenize using left-padding recommended logic
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0, # Deterministic outputs requested
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            gen_len = inputs.input_ids.shape[-1]
            response = tokenizer.decode(outputs[0][gen_len:], skip_special_tokens=True)
        else:
            # Provide a dummy response mimicking the correct format
            response = f"Let's think step by step. First we parse the question... thus the answer is \\boxed{{{item['ground_truth']}}}."
            
        pred_ans = extract_boxed_answer(response)
        
        predictions.append(pred_ans)
        references.append(item["ground_truth"])
        
        print(f"\nQ: {item['question']}")
        print(f"Ref: {item['ground_truth']}")
        print(f"Pred (Extracted): {pred_ans}")
        
    acc = compute_accuracy(predictions, references)
    print(f"\n==============================")
    print(f"Baseline Accuracy: {acc * 100:.2f}%")
    print(f"Total Samples: {len(dataset)}")
    print(f"==============================")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Using a smaller placeholder model or the specified Nemotron path if downloaded.
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Mini-4B-Instruct") 
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    
    run_baseline(args.model, args.samples)
