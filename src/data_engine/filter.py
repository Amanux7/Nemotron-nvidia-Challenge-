def filter_high_quality(samples: list) -> list:
    """
    Filters synthetic data: removes generic errors or malformed ground truths.
    For training, we only want high-confidence or syntactically valid data.
    """
    filtered = []
    for s in samples:
        gt = s.get("ground_truth", "")
        # Very basic check: Is the ground truth numeric?
        # In a real pipeline, we would evaluate the reasoning steps for logical consistency here.
        if isinstance(gt, str) and gt.lstrip('-').replace('.','',1).isdigit():
            filtered.append(s)
            
    print(f"Filtered {len(samples) - len(filtered)} irregular samples out of {len(samples)}.")
    return filtered
