"""
Fine-Tune Nemotron-30B with CoT Data (QLoRA)
==============================================
Replaces the answer-only training with Chain-of-Thought training.
Designed for Kaggle RTX PRO 6000 Blackwell (95GB VRAM).

This is a PYTHON SCRIPT version. See train_nemotron_cot_notebook.py for
the notebook-cell-ready version you can paste into Kaggle.

Usage:
    python train_nemotron_cot.py \
        --cot_data train_cot_clean.jsonl \
        --model_path /kaggle/input/models/.../nemotron-3-nano-30b-... \
        --output_dir nvidia_nemotron_lora_cot \
        --epochs 2 \
        --lr 1e-5 \
        --lora_rank 64
"""

import argparse
import json
import os
import sys
import gc

# Suppress TensorFlow/XLA initialization warnings (fixes cuFFT/cuDNN/computation_placer ALREADY REGISTERED errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_templates import format_training_example


def load_cot_dataset(cot_path: str) -> list:
    """Load the CoT JSONL and format for training."""
    records = []
    with open(cot_path, 'r') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                text = format_training_example(
                    prompt=obj['prompt'],
                    reasoning_trace=obj['reasoning_trace'],
                    answer=obj['answer'],
                )
                records.append({"text": text, "id": obj.get("id", "")})
    print(f"📊 Loaded {len(records)} CoT training examples")
    return records


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron with CoT data")
    parser.add_argument("--cot_data", type=str, required=True, help="Path to train_cot_clean.jsonl")
    parser.add_argument("--model_path", type=str,
                        default="/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
                        help="Path to base Nemotron model")
    parser.add_argument("--output_dir", type=str, default="nvidia_nemotron_lora_cot")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--eval_split", type=float, default=0.10, help="Fraction for validation")
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--use_unsloth", action="store_true", default=True, help="Use Unsloth for faster loading")
    args = parser.parse_args()
    
    import torch
    from datasets import Dataset
    from transformers import (
        Trainer, TrainingArguments, DataCollatorForSeq2Seq,
        EarlyStoppingCallback, AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    
    # ================================================================
    # 1. Load base model
    # ================================================================
    gc.collect()
    torch.cuda.empty_cache()
    
    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            print("⏳ Loading Nemotron via Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_path,
                max_seq_length=args.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                trust_remote_code=True,
            )
            print("✅ Model loaded via Unsloth")
        except ImportError:
            args.use_unsloth = False
            print("⚠️ Unsloth not available, falling back to native HF")
    
    if not args.use_unsloth:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ================================================================
    # 2. Attach LoRA (r=64 for reasoning transfer)
    # ================================================================
    if not isinstance(model, PeftModel):
        print("🧠 Attaching LoRA adapter...")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,  # Small dropout for regularization
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✅ LoRA attached: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")
    
    # ================================================================
    # 3. Apply MoE fix (same as user's notebook)
    # ================================================================
    def apply_moe_patch(model):
        for module in model.modules():
            if hasattr(module.__class__, "moe") and not hasattr(module.__class__, "_is_cot_patched"):
                def fixed_moe(self, hidden_states, topk_indices, topk_weights):
                    orig_shape = hidden_states.shape
                    topk_weights = topk_weights.to(hidden_states.dtype)
                    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                    final_hidden_states = torch.zeros_like(hidden_states)
                    for expert_idx, expert in enumerate(self.experts):
                        token_indices = (topk_indices == expert_idx).nonzero(as_tuple=True)[0]
                        if token_indices.numel() > 0:
                            expert_input = hidden_states[token_indices]
                            expert_weights = topk_weights[token_indices, (topk_indices == expert_idx).nonzero(as_tuple=True)[1]]
                            expert_output = expert(expert_input)
                            final_hidden_states.index_add_(0, token_indices, (expert_output * expert_weights.unsqueeze(-1)).to(final_hidden_states.dtype))
                        else:
                            dummy_input = torch.zeros((1, hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype)
                            dummy_out = expert(dummy_input)
                            final_hidden_states = final_hidden_states + (dummy_out * 0.0).sum()
                    return final_hidden_states.view(*orig_shape)
                module.__class__.moe = fixed_moe
                module.__class__._is_cot_patched = True
                print("✅ MoE Router patched (Quantization-Safe)")
                break
    
    apply_moe_patch(model)
    
    # Final dtype alignment
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
    
    # ================================================================
    # 4. Prepare dataset with -100 masking on prompt tokens
    # ================================================================
    records = load_cot_dataset(args.cot_data)
    
    RESPONSE_TEMPLATE = "### Assistant:\n"
    
    def tokenize_and_mask(example):
        text = example["text"]
        
        if RESPONSE_TEMPLATE in text:
            prompt_with_template = text.split(RESPONSE_TEMPLATE)[0] + RESPONSE_TEMPLATE
        else:
            prompt_with_template = text
        
        tokenized_full = tokenizer(text, truncation=True, max_length=args.max_seq_length)
        tokenized_prompt = tokenizer(prompt_with_template, truncation=True, max_length=args.max_seq_length)
        
        input_ids = tokenized_full["input_ids"]
        labels = input_ids.copy()
        
        # Mask prompt tokens with -100
        prompt_len = len(tokenized_prompt["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized_full["attention_mask"],
            "labels": labels,
        }
    
    dataset = Dataset.from_list(records)
    
    # Train/val split
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"📊 Split: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    print("✂️ Tokenizing and applying -100 masking...")
    train_dataset = train_dataset.map(tokenize_and_mask, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(tokenize_and_mask, remove_columns=eval_dataset.column_names)
    print("✅ Tokenization complete")
    
    # ================================================================
    # 5. Training with early stopping
    # ================================================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        bf16=True,
        logging_steps=5,
        optim="adamw_8bit",
        report_to="none",
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
    )
    
    callbacks = []
    if eval_dataset and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
        args=training_args,
        callbacks=callbacks,
    )
    
    print(f"\n🚀 Starting CoT fine-tuning!")
    print(f"   Epochs: {args.epochs}")
    print(f"   LR: {args.lr}")
    print(f"   LoRA rank: {args.lora_rank}")
    print(f"   Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"   Max seq length: {args.max_seq_length}")
    
    result = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Steps: {result.global_step}")
    print(f"   Final loss: {result.training_loss:.4f}")
    
    # ================================================================
    # 6. Save adapter
    # ================================================================
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"💾 Adapter saved to {args.output_dir}")
    
    # Create submission zip
    import zipfile
    zip_path = os.path.join(os.path.dirname(args.output_dir), "submission.zip")
    files_to_zip = [
        "adapter_config.json", "adapter_model.safetensors",
        "README.md", "special_tokens_map.json",
        "tokenizer_config.json", "tokenizer.json"
    ]
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for fname in files_to_zip:
            fpath = os.path.join(args.output_dir, fname)
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=fname)
    print(f"📦 Submission zip: {zip_path}")


if __name__ == "__main__":
    main()
