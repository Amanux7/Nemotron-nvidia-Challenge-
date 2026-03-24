def reasoning_trace_distillation(candidate_traces: list) -> str:
    """
    Optimization Technique 1: Reasoning Trace Distillation.
    When multiple execution paths output the correct majority vote, 
    this selects the trace with highest brevity and directness (Occam's Razor principle),
    preventing the model from learning meandering, inefficient thought patterns.
    """
    if not candidate_traces:
        return ""
    # Sort candidates emphasizing the shortest, most concise trace path
    candidate_traces.sort(key=len)
    return candidate_traces[0]

def dynamic_quality_scoring(sample: dict) -> float:
    """
    Optimization Technique 2: Dynamic Data Quality Scoring System.
    Assigns a normalized weight (0.0 - 1.0) to a training sample based on 
    the richness of its reasoning trace structural milestones.
    This acts as a continuous quality threshold for filtering.
    """
    trace = sample.get("reasoning_trace", "")
    score = 0.0
    
    # Reward structural logic transition markers
    if "first" in trace.lower() or "step 1" in trace.lower(): score += 0.2
    if "therefore" in trace.lower() or "thus" in trace.lower(): score += 0.3
    if "\\boxed" in trace: score += 0.5
    
    # Penalize low-effort traces (hallucinations under 50 chars usually)
    if len(trace) < 50: score -= 0.5
    
    return max(0.0, min(1.0, score))
