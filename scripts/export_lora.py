import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def export_lora_to_submission(base_model_path: str, lora_adapter_path: str, output_path: str):
    """
    Merges the trained LoRA adapter back into the base model weights 
    for a standalone submission payload, removing PEFT config overhead 
    so that it can be optimally loaded into inference engines like vLLM.
    """
    print(f"Loading Base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print(f"Applying LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("Merging PEFT weights into base distribution...")
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Merge could not be executed: {e}. Are we running on CPU without proper precision?")
        return
        
    print(f"Exporting standard submission format model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("Export sequence complete. Proceed to packaging target!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Path to base model (e.g. nvidia/Nemotron-3-Nano-30B)")
    parser.add_argument("--lora", type=str, required=True, help="Path to the trained LoRA adapter weights directory")
    parser.add_argument("--out", type=str, default="submission_model", help="Target output directory for the combined model")
    args = parser.parse_args()
    
    export_lora_to_submission(args.base, args.lora, args.out)
