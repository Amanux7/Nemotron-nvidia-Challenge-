"""
===========================================================================
KAGGLE NOTEBOOK: Training + Submission (RTX PRO 6000 Blackwell)
===========================================================================
This is the MAIN notebook that runs on your Kaggle RTX PRO 6000.
It does two things:
  Phase A: Fine-tune Nemotron-30B with CoT data (QLoRA)
  Phase B: Run inference with self-consistency and submit

PREREQUISITES:
  1. Run kaggle_cot_generation.py on a separate free T4 notebook
  2. Upload the output train_cot.jsonl as a Kaggle Dataset
  3. Attach that dataset + the Nemotron model + Unsloth wheels to this notebook
===========================================================================
"""

# %%
# =====================================================
# CELL 1: SAFE OFFLINE INSTALLER (same as your original)
# =====================================================
import sys, subprocess, os, importlib, site

# Suppress TensorFlow/XLA initialization warnings (fixes cuFFT/cuDNN/computation_placer ALREADY REGISTERED errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("🧹 Installing Unsloth while PROTECTING Kaggle's native environment...")

wheel_files = []
toxic_packages = ["torch-", "torch==", "torchvision", "torchaudio", "numpy", "scipy", "pillow"]

for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".whl") or f.endswith(".tar.gz"):
            if any(toxic in f.lower() for toxic in toxic_packages):
                continue
            wheel_files.append(os.path.join(root, f))

print(f"📦 Installing {len(wheel_files)} offline AI packages safely...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"] + wheel_files)
importlib.reload(site)
print("✅ Safe install complete.")


# %%
# =====================================================
# CELL 2: LOAD COT DATASET (from your generated JSONL)
# =====================================================
import pandas as pd
import json
import re
from datasets import Dataset

# === CONFIGURE: Point to your uploaded CoT dataset ===
COT_DATA_PATH = None  # Will auto-discover
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith("_clean.jsonl") or f == "train_cot.jsonl":
            COT_DATA_PATH = os.path.join(root, f)
            break
    if COT_DATA_PATH:
        break

if not COT_DATA_PATH:
    # Fallback: search for any .jsonl
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".jsonl"):
                COT_DATA_PATH = os.path.join(root, f)
                break
        if COT_DATA_PATH:
            break

if not COT_DATA_PATH:
    raise FileNotFoundError("❌ Could not find CoT JSONL in /kaggle/input/")

print(f"✅ Found CoT data: {COT_DATA_PATH}")

# Load and format
RESPONSE_TEMPLATE = "### Assistant:\n"

records = []
with open(COT_DATA_PATH) as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            text = (
                f"User: Solve this reasoning task step-by-step.\n"
                f"Task: {obj['prompt']}\n\n"
                f"{RESPONSE_TEMPLATE}"
                f"<reasoning>\n"
                f"{obj['reasoning_trace']}\n"
                f"</reasoning>\n\n"
                f"\\boxed{{{obj['answer']}}}"
            )
            records.append({"text": text})

print(f"📊 Loaded {len(records)} CoT training examples")

# Create train/eval split
dataset = Dataset.from_list(records)
split = dataset.train_test_split(test_size=0.10, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"📊 Split: {len(train_dataset)} train / {len(eval_dataset)} eval")


# %%
# =====================================================
# CELL 3: LOAD NEMOTRON MODEL
# =====================================================
import importlib.util

# Cloak torchvision
real_find_spec = importlib.util.find_spec
def hooked_find_spec(name, package=None):
    if "torchvision" in name: return None
    return real_find_spec(name, package)
importlib.util.find_spec = hooked_find_spec

from unsloth import FastLanguageModel
import torch

print("⏳ Loading Nemotron-30B...")

MODEL_ID = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,   # 👈 REDUCED back to 2048 to drastically speed up training
    dtype=torch.bfloat16,
    load_in_4bit=True,
    trust_remote_code=True,
)

print("✅ Model loaded")


# %%
# =====================================================
# CELL 4: ATTACH LORA + TRAIN WITH COT DATA
# =====================================================
import gc
import shutil
import os
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# --- 🧹 CRITICAL FIX: Clear old Kaggle outputs before starting! ---
print("🧹 Cleaning up old checkpoints from disk to prevent Kaggle 20GB limit crash...")
for path in ["outputs_cot", "nvidia_nemotron_lora_cot", "submission.zip"]:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
print("✅ Disk cleanup complete!")

gc.collect()
torch.cuda.empty_cache()

# --- LoRA Configuration (UPGRADED for reasoning) ---
if not isinstance(model, PeftModel):
    print("🧠 Attaching LoRA (r=64 for reasoning transfer)...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    lora_config = LoraConfig(
        r=64,              # 👈 UPGRADED from 16 (reasoning needs higher rank)
        lora_alpha=128,    # 👈 2x rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, # 👈 Small regularization
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA attached: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

# --- MoE Fix (same as your original) ---
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
            print("✅ MoE Router patched")
            break

apply_moe_patch(model)

for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.bfloat16)

# --- Tokenize with -100 masking ---
def tokenize_and_mask(example):
    text = example["text"]
    if RESPONSE_TEMPLATE in text:
        prompt_part = text.split(RESPONSE_TEMPLATE)[0] + RESPONSE_TEMPLATE
    else:
        prompt_part = text
    
    tokenized_full = tokenizer(text, truncation=True, max_length=2048)
    tokenized_prompt = tokenizer(prompt_part, truncation=True, max_length=2048)
    
    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()
    
    # MASK prompt tokens with -100 (model only learns reasoning + answer)
    prompt_len = len(tokenized_prompt["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    
    return {"input_ids": input_ids, "attention_mask": tokenized_full["attention_mask"], "labels": labels}

print("✂️ Tokenizing with -100 masking...")
tok_train = train_dataset.map(tokenize_and_mask, remove_columns=train_dataset.column_names)
tok_eval = eval_dataset.map(tokenize_and_mask, remove_columns=eval_dataset.column_names)
print("✅ Done")

# --- Train ---
trainer = Trainer(
    model=model,
    train_dataset=tok_train,
    eval_dataset=tok_eval,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
    args=TrainingArguments(
        output_dir="outputs_cot",
        per_device_train_batch_size=4,  # 👈 4 parallel sequences (utilizes 48GB VRAM)
        gradient_accumulation_steps=2,  # 👈 4x2 = 8 effective batch size (same as before)
        num_train_epochs=1.0,           # 👈 CUT IN HALF (1 epoch is usually enough for LoRA)
        learning_rate=1e-5,             # 👈 GENTLER (was 2e-5)
        lr_scheduler_type="cosine",     # 👈 Cosine decay
        warmup_ratio=0.10,
        bf16=True,
        logging_steps=5,
        optim="adamw_8bit",
        report_to="none",
        save_strategy="no",             # 👈 BULLETPROOF FIX: Prevent 100% of out-of-disk crashes
        eval_strategy="steps",
        eval_steps=25,
        weight_decay=0.01,
        max_grad_norm=1.0,
    ),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Must be removed if save_strategy="no"
)

print("🚀 Starting CoT fine-tuning (Batch=4, Acc=2, r=64, lr=1e-5, cosine)...")
result = trainer.train()
print(f"✅ Done! Steps: {result.global_step}, Loss: {result.training_loss:.4f}")


# %%
# =====================================================
# CELL 5: SAVE ADAPTER + CREATE SUBMISSION ZIP
# =====================================================
import zipfile
from IPython.display import FileLink

output_dir = "nvidia_nemotron_lora_cot"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

submission_zip = "submission.zip"
files_to_include = [
    "adapter_config.json", "adapter_model.safetensors",
    "README.md", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json"
]

with zipfile.ZipFile(submission_zip, 'w') as zipf:
    for f in files_to_include:
        fp = os.path.join(output_dir, f)
        if os.path.exists(fp):
            zipf.write(fp, arcname=f)
            print(f"📦 {f}")

print(f"\n🚀 {submission_zip} ready!")
FileLink(submission_zip)
