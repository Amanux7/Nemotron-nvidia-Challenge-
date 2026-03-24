import random

def generate_synthetic_math_problems(num_samples: int):
    """
    Generates an expanded set of multi-step arithmetic, logic, and symbolic templates.
    Produces diverse seed problems to feed the self-refinement engine.
    """
    templates = [
        # Arithmetic Multi-step
        {"temp": "Calculate (({a} + {b}) * {c}) - {d}.", "type": "arithmetic", "diff": "medium", "gt_func": lambda a,b,c,d: str(((a+b)*c)-d)},
        {"temp": "If a factory produces {a} widgets a day, and {b} are defective, how many functional widgets are made in {c} days?", "type": "arithmetic", "diff": "medium", "gt_func": lambda a,b,c,d: str((a-b)*c)},
        # Logic
        {"temp": "If A is taller than B, and B is taller than C. Is C taller than A? Answer Yes or No inside the box.", "type": "logic", "diff": "easy", "gt_func": lambda a,b,c,d: "No"},
        {"temp": "John has {a} apples. He gives {b} to Mary. Mary eats {c}. How many apples does Mary have left?", "type": "logic", "diff": "easy", "gt_func": lambda a,b,c,d: str(b-c) if b>=c else "0"},
        # Symbolic
        {"temp": "Simplify the algebraic expression: {a}x + {b}x - {c}y + {d}y. Represent in alphabetical order.", "type": "symbolic", "diff": "hard", "gt_func": lambda a,b,c,d: f"{a+b}x + {d-c}y" if d-c >= 0 else f"{a+b}x - {c-d}y"},
        {"temp": "Solve for x given x > 0: {a}x^2 = {a} * {b}^2.", "type": "symbolic", "diff": "hard", "gt_func": lambda a,b,c,d: str(b)}
    ]
    
    samples = []
    for _ in range(num_samples):
        choice = random.choice(templates)
        t = choice["temp"]
        
        # Ensure non-trivial params
        a, b = random.randint(5, 50), random.randint(2, 20)
        c, d = random.randint(1, 15), random.randint(1, 15)
        
        # Avoid negative widget production or negative apple transfer edge cases simply
        if choice["type"] == "arithmetic" and "widgets" in t:
            if b >= a: a = b + random.randint(1, 10)
        if choice["type"] == "logic" and "apples" in t:
            if c > b: b = c + random.randint(0, 5)
            
        q = t.format(a=a, b=b, c=c, d=d)
        gt = choice["gt_func"](a, b, c, d)
        
        samples.append({
            "question": q,
            "difficulty": choice["diff"],
            "type": choice["type"],
            "ground_truth": gt
        })
            
    return samples

if __name__ == "__main__":
    print("Test Sample Generation:")
    data = generate_synthetic_math_problems(5)
    for d in data:
        print(f"[{d['type'].upper()}] {d['question']} | GT: {d['ground_truth']}")
