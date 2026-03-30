"""
Synthetic CoT Data Generator for NVIDIA Nemotron Reasoning Challenge
====================================================================
Generates step-by-step reasoning traces using a teacher model (DeepSeek-R1-Distill-Qwen-7B)
with vLLM for high-throughput inference + rejection sampling.

Designed for Kaggle free-tier T4 GPU (16GB VRAM).

Usage on Kaggle:
    python generate_cot_data.py \
        --train_csv /kaggle/input/.../train.csv \
        --output_file /kaggle/working/train_cot.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --num_samples 8 \
        --batch_size 64 \
        --temperature 0.7
"""

import argparse
import json
import os
import sys
import re
import time
import pandas as pd
from pathlib import Path

# ============================================================================
# Attempt vLLM import; fallback to HuggingFace transformers if unavailable
# ============================================================================
USE_VLLM = True
try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM loaded — high-throughput mode enabled")
except ImportError:
    USE_VLLM = False
    print("⚠️ vLLM not available, falling back to HuggingFace transformers (slower)")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ============================================================================
# Import our shared templates
# ============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_templates import (
    TEACHER_SYSTEM_PROMPT,
    detect_category,
    build_teacher_prompt,
    extract_boxed_answer,
    answers_match,
)


def load_train_data(csv_path: str) -> pd.DataFrame:
    """Load train.csv and validate columns."""
    df = pd.read_csv(csv_path)
    
    # Auto-detect columns
    q_candidates = ['question', 'problem', 'prompt', 'instruction', 'text']
    a_candidates = ['answer', 'solution', 'response', 'output', 'target', 'expected_answer']
    
    cols = df.columns.tolist()
    q_col = next((c for c in cols if c.lower() in q_candidates), cols[0])
    a_col = next((c for c in cols if c.lower() in a_candidates), cols[1] if len(cols) > 1 else cols[-1])
    
    df = df.rename(columns={q_col: 'prompt', a_col: 'answer'})
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    df['answer'] = df['answer'].astype(str).str.strip()
    df['category'] = df['prompt'].apply(detect_category)
    
    print(f"📊 Loaded {len(df)} rows from {csv_path}")
    print(f"   Category distribution:")
    for cat, count in df['category'].value_counts().items():
        print(f"     {cat}: {count}")
    
    return df


# ============================================================================
# vLLM-based generation (HIGH THROUGHPUT)
# ============================================================================
class VLLMGenerator:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.90):
        print(f"🚀 Loading teacher model via vLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="half",
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=42,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("✅ Teacher model loaded via vLLM")
    
    def generate_batch(self, prompts: list, num_samples: int, temperature: float, max_tokens: int):
        """Generate multiple samples per prompt in one batch call."""
        sampling_params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            stop=["User:", "\n\nUser:"],
        )
        
        # Build chat-format prompts
        chat_prompts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback for models without chat template
                text = f"System: {TEACHER_SYSTEM_PROMPT}\n\nUser: {p}\n\nAssistant:"
            chat_prompts.append(text)
        
        outputs = self.llm.generate(chat_prompts, sampling_params)
        
        # Organize: list of list of strings
        results = []
        for output in outputs:
            samples = [o.text for o in output.outputs]
            results.append(samples)
        return results


# ============================================================================
# HuggingFace Transformers fallback (SLOWER but works without vLLM)
# ============================================================================
class HFGenerator:
    def __init__(self, model_name: str):
        print(f"🚀 Loading teacher model via HuggingFace: {model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print("✅ Teacher model loaded via HuggingFace (4-bit)")
    
    def generate_batch(self, prompts: list, num_samples: int, temperature: float, max_tokens: int):
        """Generate samples one prompt at a time (HF doesn't batch as efficiently)."""
        all_results = []
        
        for p in prompts:
            messages = [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = f"System: {TEACHER_SYSTEM_PROMPT}\n\nUser: {p}\n\nAssistant:"
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            samples = []
            for _ in range(num_samples):
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                generated = self.tokenizer.decode(
                    output[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                samples.append(generated)
            
            all_results.append(samples)
        
        return all_results


# ============================================================================
# REJECTION SAMPLING — only keep traces with correct answers
# ============================================================================
def rejection_sample(
    row_id: str,
    prompt: str,
    ground_truth: str,
    category: str,
    generated_samples: list,
) -> dict:
    """
    From N generated samples, find the best one that:
    1. Contains \\boxed{answer}
    2. The extracted answer matches ground truth
    3. Among correct traces, pick the shortest (highest reasoning density)
    """
    correct_traces = []
    
    for i, sample in enumerate(generated_samples):
        extracted = extract_boxed_answer(sample)
        if extracted is None:
            continue
        
        if answers_match(extracted, ground_truth):
            # Clean up the reasoning trace
            # Remove everything after the last \boxed{} for cleanliness
            boxed_pattern = r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            last_boxed = list(re.finditer(boxed_pattern, sample))
            if last_boxed:
                clean_trace = sample[:last_boxed[-1].end()].strip()
            else:
                clean_trace = sample.strip()
            
            correct_traces.append({
                "trace": clean_trace,
                "extracted_answer": extracted,
                "length": len(clean_trace),
                "sample_idx": i,
            })
    
    if not correct_traces:
        return None
    
    # Pick the shortest correct trace (highest reasoning density)
    best = min(correct_traces, key=lambda x: x["length"])
    
    # Extract just the reasoning part (everything before \boxed{})
    reasoning_only = re.split(r'\\boxed\{', best["trace"])[0].strip()
    
    return {
        "id": row_id,
        "prompt": prompt,
        "category": category,
        "reasoning_trace": reasoning_only,
        "answer": ground_truth,
        "full_trace": best["trace"],
        "num_correct": len(correct_traces),
        "num_total": len(generated_samples),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CoT data")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--output_file", type=str, default="train_cot.jsonl", help="Output JSONL path")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Teacher model name/path")
    parser.add_argument("--num_samples", type=int, default=8, help="Candidates per problem (N for rejection sampling)")
    parser.add_argument("--batch_size", type=int, default=32, help="Prompts per batch")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument("--gpu_memory", type=float, default=0.90, help="GPU memory utilization for vLLM")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()
    
    # Load data
    df = load_train_data(args.train_csv)
    
    # Check for resume
    completed_ids = set()
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                obj = json.loads(line)
                completed_ids.add(str(obj['id']))
        print(f"📂 Resuming: {len(completed_ids)} already completed")
    
    # Filter out completed
    df_todo = df[~df['id'].astype(str).isin(completed_ids)].reset_index(drop=True)
    print(f"🎯 {len(df_todo)} rows to process")
    
    if len(df_todo) == 0:
        print("✅ All rows already processed!")
        return
    
    # Initialize generator
    if USE_VLLM:
        generator = VLLMGenerator(args.model, args.gpu_memory)
    else:
        generator = HFGenerator(args.model)
    
    # Process in batches
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    # Open output file in append mode
    mode = 'a' if args.resume else 'w'
    output_fh = open(args.output_file, mode)
    
    # Also save failures for analysis
    fail_file = args.output_file.replace('.jsonl', '_failures.jsonl')
    fail_fh = open(fail_file, mode)
    
    total_batches = (len(df_todo) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(df_todo))
        batch_df = df_todo.iloc[batch_start:batch_end]
        
        # Build teacher prompts
        teacher_prompts = []
        for _, row in batch_df.iterrows():
            tp = build_teacher_prompt(row['prompt'], row['category'])
            teacher_prompts.append(tp)
        
        # Generate
        batch_time = time.time()
        try:
            all_samples = generator.generate_batch(
                teacher_prompts, args.num_samples, args.temperature, args.max_tokens
            )
        except Exception as e:
            print(f"❌ Batch {batch_idx+1}/{total_batches} failed: {e}")
            continue
        
        # Rejection sample
        for i, (_, row) in enumerate(batch_df.iterrows()):
            result = rejection_sample(
                row_id=str(row['id']),
                prompt=row['prompt'],
                ground_truth=str(row['answer']),
                category=row['category'],
                generated_samples=all_samples[i],
            )
            
            if result:
                output_fh.write(json.dumps(result) + '\n')
                output_fh.flush()
                success_count += 1
            else:
                fail_record = {
                    "id": str(row['id']),
                    "prompt": row['prompt'][:200],
                    "ground_truth": str(row['answer']),
                    "category": row['category'],
                    "num_samples": len(all_samples[i]),
                    "sample_previews": [s[:200] for s in all_samples[i][:2]],
                }
                fail_fh.write(json.dumps(fail_record) + '\n')
                fail_fh.flush()
                fail_count += 1
        
        elapsed = time.time() - batch_time
        total_elapsed = time.time() - start_time
        processed = batch_end
        rate = processed / total_elapsed * 3600 if total_elapsed > 0 else 0
        
        print(
            f"📦 Batch {batch_idx+1}/{total_batches} | "
            f"✅ {success_count} | ❌ {fail_count} | "
            f"Success rate: {success_count/(success_count+fail_count)*100:.1f}% | "
            f"Speed: {rate:.0f} rows/hr | "
            f"Batch time: {elapsed:.1f}s"
        )
    
    output_fh.close()
    fail_fh.close()
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 GENERATION COMPLETE")
    print(f"   ✅ Success: {success_count}/{success_count+fail_count} ({success_count/(success_count+fail_count)*100:.1f}%)")
    print(f"   ❌ Failed:  {fail_count}")
    print(f"   ⏱️  Total time: {total_elapsed/60:.1f} minutes")
    print(f"   📄 Output: {args.output_file}")
    print(f"   📄 Failures: {fail_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
