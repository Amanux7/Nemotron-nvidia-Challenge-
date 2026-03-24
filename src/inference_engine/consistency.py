import sys, os
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from evaluation_engine.extractor import extract_boxed_answer

def generate_with_consistency(prompt: str, model_generate_fn, num_samples: int = 3, max_retries: int = 2):
    """
    Inference mechanism wrapping model output logic directly resolving constraint formatting requirements.
    - Operates Self-Consistency voting arrays dynamically targeting `num_samples` output predictions.
    - Implements a retry constraint layer executing loop overrides up to `max_retries` if \\boxed formats drop.
    """
    answers = []
    valid_traces = []
    
    for _ in range(num_samples):
        valid = False
        trace = ""
        # Format Execution Constraints Retries
        for retry in range(max_retries + 1):
            trace = model_generate_fn(prompt)
            if "\\boxed{" in trace:
                valid = True
                break
                
        if valid:
            ans = extract_boxed_answer(trace)
            answers.append(ans)
            valid_traces.append(trace)
            
    if not answers:
        return "", ""
        
    # Majority Vote Extraction execution across collected structural parameters
    most_frequent = Counter(answers).most_common(1)[0][0]
    best_trace = next(t for t, a in zip(valid_traces, answers) if a == most_frequent)
    
    return most_frequent, best_trace
