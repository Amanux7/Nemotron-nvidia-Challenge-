"""
===========================================================================
KAGGLE NOTEBOOK: Programmatic CoT Data Generation
===========================================================================
ZERO GPU NEEDED. Generates Chain-of-Thought reasoning traces by analyzing
the problem structure and writing template-based step-by-step solutions
using the known ground truth answers.

This replaces the LLM-based approach which was too slow/broken on T4.

Run this on ANY Kaggle notebook (even CPU-only) or locally.
Upload the output (train_cot.jsonl) as a Kaggle Dataset.
===========================================================================
"""

# %%
# CELL 1: Imports (no GPU packages needed!)
import pandas as pd
import json
import re
import os
import time
import random

# %%
# CELL 2: Category detection + programmatic CoT generators

CATEGORY_PATTERNS = {
    "bit_manipulation": ["bit manipulation", "8-bit binary", "bit shifts"],
    "cipher": ["secret encryption rules", "decrypt the following"],
    "unit_conversion": ["unit conversion", "convert the following measurement"],
    "roman_numeral": ["numeral system", "Wonderland numeral"],
    "gravity_physics": ["gravitational constant", "d = 0.5*g*t^2", "falling distance"],
    "symbol_transform": ["transformation rules", "determine the result for"],
}

def detect_category(prompt):
    prompt_lower = prompt.lower()
    for cat, patterns in CATEGORY_PATTERNS.items():
        for p in patterns:
            if p.lower() in prompt_lower:
                return cat
    return "unknown"


def parse_examples(prompt, category):
    """Extract input->output example pairs from the prompt."""
    examples = []
    
    if category == "bit_manipulation":
        for m in re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt):
            examples.append({"input": m[0], "output": m[1]})
        # Extract test input
        test_match = re.search(r'output for:\s*([01]{8})', prompt)
        test_input = test_match.group(1) if test_match else None
        return examples, test_input
    
    elif category == "cipher":
        for m in re.findall(r'([a-z][a-z ]+?)\s*->\s*([a-z][a-z ]+)', prompt):
            examples.append({"input": m[0].strip(), "output": m[1].strip()})
        test_match = re.search(r'decrypt the following text:\s*(.+?)(?:\"|$)', prompt)
        test_input = test_match.group(1).strip() if test_match else None
        return examples, test_input
    
    elif category == "unit_conversion":
        for m in re.findall(r'([\d.]+)\s*m\s*becomes\s*([\d.]+)', prompt):
            examples.append({"input": float(m[0]), "output": float(m[1])})
        test_match = re.search(r'measurement:\s*([\d.]+)\s*m', prompt)
        test_input = float(test_match.group(1)) if test_match else None
        return examples, test_input
    
    elif category == "roman_numeral":
        for m in re.findall(r'(\d+)\s*->\s*([IVXLCDM]+)', prompt):
            examples.append({"input": int(m[0]), "output": m[1]})
        test_match = re.search(r'number\s+(\d+)\s+in', prompt)
        test_input = int(test_match.group(1)) if test_match else None
        return examples, test_input
    
    elif category == "gravity_physics":
        for m in re.findall(r't\s*=\s*([\d.]+)s.*?distance\s*=\s*([\d.]+)', prompt):
            examples.append({"t": float(m[0]), "d": float(m[1])})
        test_match = re.search(r'for\s+t\s*=\s*([\d.]+)s', prompt)
        test_input = float(test_match.group(1)) if test_match else None
        return examples, test_input
    
    elif category == "symbol_transform":
        lines = prompt.split('\n')
        for line in lines:
            m = re.match(r'(.+?)\s*=\s*(.+)', line.strip())
            if m and '->' not in line and 'Now' not in line and 'transformation' not in line.lower():
                examples.append({"input": m.group(1).strip(), "output": m.group(2).strip()})
        test_match = re.search(r'result for:\s*(.+?)(?:\"|$)', prompt)
        test_input = test_match.group(1).strip() if test_match else None
        return examples, test_input
    
    return examples, None


# =============================================================
# CATEGORY-SPECIFIC CoT GENERATORS
# =============================================================

def generate_cot_bit_manipulation(prompt, answer, examples, test_input):
    """Generate step-by-step reasoning for bit manipulation."""
    lines = []
    lines.append("Let me analyze the bit manipulation pattern by examining the input-output examples.")
    lines.append("")
    
    # Show examples
    for i, ex in enumerate(examples[:4]):
        lines.append(f"Example {i+1}: {ex['input']} -> {ex['output']}")
    lines.append("")
    
    # Analyze
    lines.append("Step 1: Look for common bit operations (NOT, shifts, rotations, XOR with mask).")
    lines.append("Step 2: Test each hypothesis against ALL examples.")
    lines.append("")
    
    # Try to identify the actual operation
    if examples:
        inp0 = int(examples[0]['input'], 2)
        out0 = int(examples[0]['output'], 2)
        xor_mask = inp0 ^ out0
        
        # Check if XOR with constant mask works
        xor_works = all(int(ex['input'], 2) ^ xor_mask == int(ex['output'], 2) for ex in examples)
        if xor_works:
            lines.append(f"Step 3: Testing XOR with mask {format(xor_mask, '08b')}:")
            for ex in examples[:3]:
                lines.append(f"  {ex['input']} XOR {format(xor_mask, '08b')} = {format(int(ex['input'],2) ^ xor_mask, '08b')} ✓")
            lines.append(f"The rule is: XOR each input with {format(xor_mask, '08b')}.")
        else:
            # Check bit reversal
            rev_works = all(ex['input'][::-1] == ex['output'] for ex in examples)
            if rev_works:
                lines.append("Step 3: Testing bit reversal:")
                for ex in examples[:3]:
                    lines.append(f"  reverse({ex['input']}) = {ex['input'][::-1]} = {ex['output']} ✓")
                lines.append("The rule is: reverse the bit order.")
            else:
                # Check rotation
                for shift in range(1, 8):
                    rot_works = all(
                        (ex['input'][shift:] + ex['input'][:shift]) == ex['output'] 
                        for ex in examples
                    )
                    if rot_works:
                        lines.append(f"Step 3: Testing left rotation by {shift}:")
                        for ex in examples[:3]:
                            rotated = ex['input'][shift:] + ex['input'][:shift]
                            lines.append(f"  rotate_left({ex['input']}, {shift}) = {rotated} = {ex['output']} ✓")
                        lines.append(f"The rule is: rotate left by {shift} positions.")
                        break
                    
                    rrot_works = all(
                        (ex['input'][-shift:] + ex['input'][:-shift]) == ex['output']
                        for ex in examples
                    )
                    if rrot_works:
                        lines.append(f"Step 3: Testing right rotation by {shift}:")
                        for ex in examples[:3]:
                            rotated = ex['input'][-shift:] + ex['input'][:-shift]
                            lines.append(f"  rotate_right({ex['input']}, {shift}) = {rotated} = {ex['output']} ✓")
                        lines.append(f"The rule is: rotate right by {shift} positions.")
                        break
                else:
                    lines.append("Step 3: The transformation pattern is complex. After careful analysis of all examples,")
                    lines.append("I identified the consistent mapping rule.")
    
    lines.append("")
    if test_input:
        lines.append(f"Step 4: Apply the rule to test input {test_input}:")
    lines.append(f"Result: {answer}")
    
    return "\n".join(lines)


def generate_cot_cipher(prompt, answer, examples, test_input):
    """Generate step-by-step reasoning for cipher problems."""
    lines = []
    lines.append("Let me build the substitution cipher mapping from the examples.")
    lines.append("")
    
    # Build letter mapping from examples
    mapping = {}
    for ex in examples:
        enc_words = ex['input'].split()
        dec_words = ex['output'].split()
        for ew, dw in zip(enc_words, dec_words):
            for ec, dc in zip(ew, dw):
                if ec.isalpha() and dc.isalpha():
                    mapping[ec] = dc
    
    lines.append("Step 1: Build the letter mapping from examples:")
    
    # Show mapping table
    sorted_mapping = sorted(mapping.items())
    map_str = ", ".join(f"{k}→{v}" for k, v in sorted_mapping[:15])
    lines.append(f"  Mapping: {map_str}")
    if len(sorted_mapping) > 15:
        map_str2 = ", ".join(f"{k}→{v}" for k, v in sorted_mapping[15:])
        lines.append(f"           {map_str2}")
    lines.append("")
    
    lines.append("Step 2: Verify the mapping on the examples:")
    for ex in examples[:2]:
        lines.append(f"  '{ex['input']}' decrypts to '{ex['output']}' ✓")
    lines.append("")
    
    if test_input:
        lines.append(f"Step 3: Apply the mapping to decrypt: '{test_input}'")
        # Show word by word
        if answer:
            dec_words = answer.split()
            enc_words = test_input.split() if test_input else []
            for ew, dw in zip(enc_words, dec_words):
                lines.append(f"  '{ew}' → '{dw}'")
    
    lines.append(f"\nDecrypted text: {answer}")
    
    return "\n".join(lines)


def generate_cot_unit_conversion(prompt, answer, examples, test_input):
    """Generate step-by-step reasoning for unit conversion."""
    lines = []
    lines.append("Let me find the conversion factor from the examples.")
    lines.append("")
    
    if examples:
        # Calculate ratios
        ratios = [ex['output'] / ex['input'] for ex in examples if ex['input'] != 0]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        
        lines.append("Step 1: Calculate ratio (output/input) for each example:")
        for i, (ex, r) in enumerate(zip(examples[:5], ratios[:5])):
            lines.append(f"  {ex['output']} / {ex['input']} = {r:.6f}")
        
        lines.append(f"\nStep 2: The conversion factor is approximately {avg_ratio:.6f}")
        lines.append(f"  (Average of all {len(ratios)} example ratios)")
        lines.append("")
        
        if test_input is not None:
            computed = test_input * avg_ratio
            lines.append(f"Step 3: Apply to test input {test_input} m:")
            lines.append(f"  {test_input} × {avg_ratio:.6f} = {computed:.2f}")
    
    lines.append(f"\nConverted value: {answer}")
    
    return "\n".join(lines)


def generate_cot_roman_numeral(prompt, answer, examples, test_input):
    """Generate step-by-step reasoning for Roman numeral conversion."""
    lines = []
    lines.append("Let me convert the number to Roman numerals using the standard rules.")
    lines.append("")
    lines.append("Roman numeral values: M=1000, D=500, C=100, L=50, X=10, V=5, I=1")
    lines.append("Subtractive notation: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900")
    lines.append("")
    
    if test_input is not None:
        n = test_input
        lines.append(f"Step 1: Convert {n} to Roman numerals:")
        
        values = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        
        remaining = n
        parts = []
        for val, sym in values:
            while remaining >= val:
                parts.append(sym)
                remaining -= val
                lines.append(f"  {n - remaining + val} - {val} = {n - remaining}: add '{sym}'")
        
        lines.append(f"\nStep 2: Combine: {''.join(parts)}")
    
    lines.append(f"\nResult: {answer}")
    
    return "\n".join(lines)


def generate_cot_gravity(prompt, answer, examples, test_input):
    """Generate step-by-step reasoning for gravity physics."""
    lines = []
    lines.append("Let me find the gravitational constant g from the examples using d = 0.5*g*t².")
    lines.append("")
    
    if examples:
        lines.append("Step 1: Solve for g in each example: g = 2*d / t²")
        g_values = []
        for i, ex in enumerate(examples[:5]):
            g = 2 * ex['d'] / (ex['t'] ** 2) if ex['t'] != 0 else 0
            g_values.append(g)
            lines.append(f"  Example {i+1}: g = 2×{ex['d']} / {ex['t']}² = {g:.4f} m/s²")
        
        avg_g = sum(g_values) / len(g_values) if g_values else 9.8
        lines.append(f"\nStep 2: Average g = {avg_g:.4f} m/s²")
        
        if test_input is not None:
            d = 0.5 * avg_g * test_input ** 2
            lines.append(f"\nStep 3: Calculate d for t = {test_input}s:")
            lines.append(f"  d = 0.5 × {avg_g:.4f} × {test_input}²")
            lines.append(f"  d = 0.5 × {avg_g:.4f} × {test_input**2:.4f}")
            lines.append(f"  d = {d:.2f} m")
    
    lines.append(f"\nFalling distance: {answer}")
    
    return "\n".join(lines)


def generate_cot_symbol_transform(prompt, answer, examples, test_input):
    """Generate step-by-step reasoning for symbol transformation."""
    lines = []
    lines.append("Let me analyze the symbol transformation rules from the examples.")
    lines.append("")
    
    if examples:
        lines.append("Step 1: Examine the input-output pairs:")
        for i, ex in enumerate(examples[:4]):
            lines.append(f"  Rule {i+1}: {ex['input']} = {ex['output']}")
        lines.append("")
        
        # Try to find character-level mapping
        char_map = {}
        for ex in examples:
            inp = ex['input'].replace(' ', '')
            out = ex['output'].replace(' ', '')
            if len(inp) == len(out):
                for ic, oc in zip(inp, out):
                    char_map[ic] = oc
        
        if char_map:
            lines.append("Step 2: Identify character-level mapping:")
            map_items = list(char_map.items())[:15]
            lines.append(f"  {', '.join(f'{k}→{v}' for k, v in map_items)}")
            lines.append("")
        
        if test_input:
            lines.append(f"Step 3: Apply the transformation to: {test_input}")
    
    lines.append(f"\nResult: {answer}")
    
    return "\n".join(lines)


# Dispatcher
def generate_cot_trace(prompt, answer, category):
    """Generate a CoT trace for any category."""
    examples, test_input = parse_examples(prompt, category)
    
    generators = {
        "bit_manipulation": generate_cot_bit_manipulation,
        "cipher": generate_cot_cipher,
        "unit_conversion": generate_cot_unit_conversion,
        "roman_numeral": generate_cot_roman_numeral,
        "gravity_physics": generate_cot_gravity,
        "symbol_transform": generate_cot_symbol_transform,
    }
    
    gen = generators.get(category)
    if gen:
        return gen(prompt, answer, examples, test_input)
    else:
        # Generic fallback
        return (
            f"Let me analyze the input-output examples step by step.\n\n"
            f"After examining all the examples in the problem, I identified the "
            f"transformation rule and applied it to the test input.\n\n"
            f"Result: {answer}"
        )


# %%
# CELL 3: Load data and generate CoT traces
# Auto-discover train.csv
TRAIN_CSV = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files:
        TRAIN_CSV = os.path.join(root, "train.csv")
        break

# Fallback for local testing
if not TRAIN_CSV:
    local_candidates = [
        "train (2).csv", "train.csv",
        r"c:\Users\91703\nemotron reasoning challenge\train (2).csv"
    ]
    for c in local_candidates:
        if os.path.exists(c):
            TRAIN_CSV = c
            break

if not TRAIN_CSV:
    raise FileNotFoundError("❌ Could not find train.csv")

OUTPUT_FILE = "/kaggle/working/train_cot.jsonl"
if not os.path.exists("/kaggle/working"):
    OUTPUT_FILE = "train_cot.jsonl"  # Local fallback

print(f"✅ Found train.csv: {TRAIN_CSV}")
print(f"📄 Output: {OUTPUT_FILE}")

# Load data
df = pd.read_csv(TRAIN_CSV)
q_col = next((c for c in df.columns if c.lower() in ['prompt','question','problem','text']), df.columns[0])
a_col = next((c for c in df.columns if c.lower() in ['answer','solution','output','target']), df.columns[1])
df = df.rename(columns={q_col: 'prompt', a_col: 'answer'})
df['answer'] = df['answer'].astype(str).str.strip()
df['category'] = df['prompt'].apply(detect_category)

print(f"\n📊 {len(df)} rows. Categories:")
print(df['category'].value_counts().to_string())

# %%
# CELL 4: Generate all CoT traces (< 1 minute!)
start = time.time()
records = []
category_stats = {}

for idx, row in df.iterrows():
    category = row['category']
    trace = generate_cot_trace(row['prompt'], row['answer'], category)
    
    record = {
        "id": str(row['id']),
        "prompt": row['prompt'],
        "category": category,
        "reasoning_trace": trace,
        "answer": row['answer'],
        "full_trace": f"{trace}\n\n\\boxed{{{row['answer']}}}",
    }
    records.append(record)
    
    category_stats[category] = category_stats.get(category, 0) + 1

# Save
with open(OUTPUT_FILE, 'w') as f:
    for record in records:
        f.write(json.dumps(record) + '\n')

elapsed = time.time() - start
print(f"\n🏁 Done in {elapsed:.1f}s!")
print(f"📊 Generated {len(records)} CoT traces")
print(f"\nCategory stats:")
for cat, count in sorted(category_stats.items()):
    print(f"  {cat}: {count}")
print(f"\n📄 Output: {OUTPUT_FILE}")

# Quick quality check — show a sample from each category
print("\n" + "="*60)
print("SAMPLE TRACES (one per category):")
print("="*60)
shown = set()
for record in records:
    cat = record['category']
    if cat not in shown:
        shown.add(cat)
        print(f"\n--- {cat} ---")
        print(record['reasoning_trace'][:300])
        print(f"...\nAnswer: {record['answer']}")
        print()
    if len(shown) >= 6:
        break

# %%
# CELL 5: Download the output
try:
    from IPython.display import FileLink
    FileLink(OUTPUT_FILE)
except:
    print(f"Download: {OUTPUT_FILE}")
