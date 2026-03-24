def normalize_answer(ans: str) -> str:
    """Normalize strings for more robust exact matching."""
    if ans is None:
        return ""
    # Lowercase and remove all whitespace
    ans = str(ans).strip().lower().replace(" ", "")
    # Remove some common punctuation that might wrap the answer
    ans = ans.rstrip('.')
    return ans

def is_equivalent(pred: str, ref: str) -> bool:
    """
    Checks if the predicted answer matches the reference answer.
    Currently uses normalized exact match and simple float equivalence.
    Can be expanded to use sympy for algebraic equivalence.
    """
    pred = normalize_answer(pred)
    ref = normalize_answer(ref)
    
    if pred == ref:
        return True
        
    # Attempt float equivalent comparison
    try:
        pred_val = float(pred)
        ref_val = float(ref)
        if abs(pred_val - ref_val) < 1e-6:
            return True
    except ValueError:
        pass
        
    return False

def compute_accuracy(predictions: list, references: list) -> float:
    """Computes the overall accuracy."""
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
        
    correct = sum(1 for p, r in zip(predictions, references) if is_equivalent(p, r))
    return correct / len(predictions)
