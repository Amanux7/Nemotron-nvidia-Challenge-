import pandas as pd
from datasets import Dataset

def raw_to_hf_dataset(samples: list) -> Dataset:
    """
    Converts a list of dicts with 'question' and 'reasoning_trace' into 
    a HuggingFace dataset formatted for instruction tuning.
    """
    formatted = []
    for s in samples:
        # If real reasoning trace exists use it, else fallback to ground truth mock
        response = s.get("reasoning_trace", f"The answer is \\boxed{{{s.get('ground_truth', '')}}}")
        formatted.append({
            "instruction": s["question"],
            "response": response
        })
        
    df = pd.DataFrame(formatted)
    return Dataset.from_pandas(df)
    
def format_prompt_func(example):
    """
    Applies the Chat or Instruction template to the raw prompt-response pairs.
    """
    text = f"User: {example['instruction']}\n\nAssistant: {example['response']}"
    return {"text": text}
