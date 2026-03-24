import argparse
import torch
import os

from .dataset import raw_to_hf_dataset, format_prompt_func

def train_lora_model(model_id: str, data: list, output_dir="lora_adapter"):
    """
    Trains a LoRA adapter natively matching the competition constraints.
    Max Rank <= 32. Uses quantized loaders for local machine stability.
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("Missing required libraries (peft, trl, bitsandbytes). LoRA script bypassed locally.")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
        
    print("Preparing dataset...")
    hf_dataset = raw_to_hf_dataset(data)
    hf_dataset = hf_dataset.map(format_prompt_func)
    
    print("Loading Base Model for LoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load quantized model: {e}")
        return output_dir
    
    print("Configuring LoRA parameters (Rank=32)...")
    lora_config = LoraConfig(
        r=32, # As per competition constraints
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2, # Keep low for 30B 
        gradient_accumulation_steps=8, # Simulate batch size of 16
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, 
        max_grad_norm=0.3,
        max_seq_length=1024,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        dataset_text_field="text"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        peft_config=lora_config,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    print("Executing Trainer...")
    trainer.train()
    
    print(f"Saving adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Mini-4B-Instruct")
    args = parser.parse_args()
    
    mock_data = [
        {"question": "What is 2+2?", "reasoning_trace": "We sum 2 and 2. The answer is \\boxed{4}."},
        {"question": "Solve 3x=9", "reasoning_trace": "Divide both sides by 3. 9/3=3. The answer is \\boxed{3}."}
    ]
    train_lora_model(args.model, mock_data)
