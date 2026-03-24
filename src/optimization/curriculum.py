def apply_curriculum_sorting(dataset: list) -> list:
    """
    Sorts a training dataset from 'easy' to 'hard' based on difficulty tags 
    and trace length (longer traces imply more complex logical dependencies).
    
    This ensures the model learns basic mapping first before tackling
    multi-step symbolic math.
    """
    difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
    
    def score(sample):
        diff_score = difficulty_map.get(sample.get("difficulty", "medium"), 2)
        trace_len = len(sample.get("reasoning_trace", ""))
        return (diff_score, trace_len)
        
    sorted_dataset = sorted(dataset, key=score)
    print(f"Curriculum sorted {len(sorted_dataset)} items from easy to hard.")
    return sorted_dataset
