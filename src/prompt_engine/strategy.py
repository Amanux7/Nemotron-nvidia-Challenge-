from collections import Counter
from .templates import BASELINE_COT_PROMPT

def generate_self_consistency_prompts(problem: str, num_paths: int = 5):
    """
    Returns identical prompts (multiplied) to allow temperature sampling 
    multiple reasoning paths (Self-Consistency pattern).
    """
    return [BASELINE_COT_PROMPT.format(problem=problem) for _ in range(num_paths)]

def majority_vote(answers: list) -> str:
    """
    Implements majority voting across multiple reasoning path outputs.
    """
    if not answers:
        return ""
    counts = Counter(answers)
    return counts.most_common(1)[0][0]
