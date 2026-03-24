import collections

def cluster_errors(predictions: list, references: list, questions: list) -> dict:
    """
    Analyzes evaluation outputs and clusters errors to identify common failure patterns.
    Outputs actionable categories: "Calculation Error", "Parsing Error", "Format Error".
    """
    clusters = collections.defaultdict(list)
    
    for p, r, q in zip(predictions, references, questions):
        # We only cluster on mistakes
        if p == r:
            continue
            
        p_str = str(p).strip()
        r_str = str(r).strip()
        
        # Heuristic 1: Did it fail to put the answer in the box?
        if not p_str:
            clusters["Format Error: Missing boxed output"].append(q)
        # Heuristic 2: Numeric miscalculations
        elif p_str.replace('.', '', 1).isdigit() and r_str.replace('.', '', 1).isdigit():
            clusters["Calculation Error: Numeric mismatch"].append(q)
        # Heuristic 3: Symbolic mismatch
        elif '\\' in p_str and '\\' in r_str:
            clusters["Symbolic Error: Latex mismatch"].append(q)
        else:
            # Fallback pattern
            clusters["Logical Error: Hallucination or Unknown"].append(q)
            
    print(f"Error analysis completed. Found {len(clusters)} distinct error clusters.")
    return dict(clusters)
