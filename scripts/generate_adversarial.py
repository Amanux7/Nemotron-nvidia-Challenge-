import json
import os

def run():
    print("Synthesizing Generalized Blind Execution Testing set...")
    blind = []
    for i in range(100):
        blind.append({
            "prompt": f"In modified logic scenarios, mapping rules switch polarity entirely. System logic: AND transforms to OR. Apply mapping constraints on variables #{i}.",
            "answer": f"BLIND_{i}"
        })
        
    print("Synthesizing Extreme Adversarial Deep Logic execution sequence...")
    adv = []
    for i in range(50):
        adv.append({
            "prompt": f"COMPOUND NOISE EDGECASE: Logical loop {i} is fundamentally unstable. Discard step 1 entirely, recursively trace step 2 outwards ignoring text format noise.",
            "answer": f"ADV_{i}"
        })
        
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/blind_test.json", "w", encoding='utf-8') as f:
        json.dump(blind, f, indent=2)
    with open("data/processed/adversarial_test.json", "w", encoding='utf-8') as f:
        json.dump(adv, f, indent=2)
        
    print(f"Generated {len(blind)} Blind Tests and {len(adv)} Adversarial Validation targets mapped securely.")

if __name__ == "__main__":
    run()
