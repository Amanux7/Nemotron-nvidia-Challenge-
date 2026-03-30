"""
Prompt Templates for NVIDIA Nemotron Reasoning Challenge
=========================================================
Centralized templates used for:
1. Teacher model CoT generation (generate_cot_data.py)
2. Student model fine-tuning (train_nemotron_cot.py)
3. Inference (inference_pipeline.py)
"""

import re

# ============================================================================
# CATEGORY DETECTION — auto-detect problem type from prompt text
# ============================================================================

CATEGORY_PATTERNS = {
    "bit_manipulation": [
        "bit manipulation", "8-bit binary", "bit shifts", "XOR", "AND, OR, NOT"
    ],
    "cipher": [
        "secret encryption rules", "decrypt the following"
    ],
    "unit_conversion": [
        "unit conversion", "convert the following measurement"
    ],
    "roman_numeral": [
        "numeral system", "Wonderland numeral"
    ],
    "gravity_physics": [
        "gravitational constant", "d = 0.5*g*t^2", "falling distance"
    ],
    "symbol_transform": [
        "transformation rules", "determine the result for"
    ],
}

def detect_category(prompt: str) -> str:
    """Detect which of the 6 problem categories a prompt belongs to."""
    prompt_lower = prompt.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in prompt_lower:
                return category
    return "unknown"


# ============================================================================
# TEACHER MODEL PROMPTS — category-specific meta-prompts
# ============================================================================

TEACHER_SYSTEM_PROMPT = (
    "You are an expert mathematical and logical reasoning engine. "
    "You must analyze input-output examples, reverse-engineer the hidden rule, "
    "and apply it precisely. Always show your complete reasoning step-by-step. "
    "Your final answer MUST be in \\boxed{answer} format — nothing else after the box."
)

# Category-specific hints to guide the teacher model
CATEGORY_HINTS = {
    "bit_manipulation": (
        "HINTS: The rule transforms 8-bit binary numbers. Common operations include:\n"
        "- Bitwise NOT (flip all bits)\n"
        "- Left/right rotation by N positions\n"
        "- XOR with a fixed mask\n"
        "- Combinations of shifts + XOR\n"
        "- Reverse bit order\n"
        "Try each operation on the examples to find which one consistently maps input→output.\n"
        "Test your hypothesis on ALL examples before applying to the test case."
    ),
    "cipher": (
        "HINTS: This is a substitution cipher on text. Common patterns:\n"
        "- Caesar cipher (shift each letter by N)\n"
        "- Simple letter substitution (each letter maps to exactly one other)\n"
        "- The shift may be different per position or per word\n"
        "Build the substitution table from the examples, then apply it to decrypt."
    ),
    "unit_conversion": (
        "HINTS: This is a linear unit conversion: output = input × constant.\n"
        "- Divide any output by its input to find the conversion factor\n"
        "- Verify the factor is consistent across all examples\n"
        "- Apply: answer = test_input × factor, rounded to 2 decimal places"
    ),
    "roman_numeral": (
        "HINTS: Convert the given integer to Roman numerals.\n"
        "- I=1, V=5, X=10, L=50, C=100, D=500, M=1000\n"
        "- Subtractive notation: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900\n"
        "- Build from largest to smallest"
    ),
    "gravity_physics": (
        "HINTS: Find the gravitational constant g from examples using d = 0.5*g*t².\n"
        "- Rearrange: g = 2*d / t²\n"
        "- Compute g from each example, take the average\n"
        "- Apply: answer = 0.5 * g_avg * t_test², rounded to 2 decimal places"
    ),
    "symbol_transform": (
        "HINTS: Find the transformation rule applied to symbol/equation strings.\n"
        "- Look for character-by-character mappings\n"
        "- Check if it's a substitution, reversal, or positional operation\n"
        "- The rule might involve arithmetic on character positions/ASCII values\n"
        "- Verify on all examples before applying"
    ),
    "unknown": (
        "HINTS: Analyze the input-output examples carefully.\n"
        "- Look for numerical, logical, or pattern-based relationships\n"
        "- Test hypotheses against ALL examples\n"
        "- Apply the discovered rule precisely to the test case"
    ),
}

def build_teacher_prompt(prompt: str, category: str = None) -> str:
    """Build the full teacher model prompt with category-specific hints."""
    if category is None:
        category = detect_category(prompt)
    
    hints = CATEGORY_HINTS.get(category, CATEGORY_HINTS["unknown"])
    
    user_message = (
        f"{hints}\n\n"
        f"PROBLEM:\n{prompt}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Analyze each input→output example pair\n"
        f"2. State the EXACT rule you discovered\n"
        f"3. Apply the rule step-by-step to the test input\n"
        f"4. Verify your answer against the examples if possible\n"
        f"5. Write your final answer as \\boxed{{answer}}\n\n"
        f"Show your complete reasoning:"
    )
    return user_message


# ============================================================================
# STUDENT MODEL TEMPLATES — for training & inference
# ============================================================================

def format_training_example(prompt: str, reasoning_trace: str, answer: str) -> str:
    """Format a single training example with CoT for the student model."""
    return (
        f"User: Solve this reasoning task step-by-step.\n"
        f"Task: {prompt}\n\n"
        f"### Assistant:\n"
        f"<reasoning>\n"
        f"{reasoning_trace}\n"
        f"</reasoning>\n\n"
        f"\\boxed{{{answer}}}"
    )


def format_inference_prompt(prompt: str) -> str:
    """Format a test prompt for inference (no answer, model generates reasoning + answer)."""
    return (
        f"User: Solve this reasoning task step-by-step.\n"
        f"Task: {prompt}\n\n"
        f"### Assistant:\n"
        f"<reasoning>\n"
    )


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================

def extract_boxed_answer(text: str) -> str:
    """Extract the answer from \\boxed{...} in model output."""
    # Try nested braces first
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Last match (final answer)
    
    # Fallback: simple pattern
    simple = re.findall(r'\\boxed\{(.+?)\}', text)
    if simple:
        return simple[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison (strips whitespace, normalizes numbers)."""
    if answer is None:
        return ""
    answer = answer.strip()
    # Try to normalize decimal numbers
    try:
        num = float(answer)
        # Round to 2 decimal places for comparison
        return f"{num:.2f}".rstrip('0').rstrip('.')
    except (ValueError, TypeError):
        pass
    return answer


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if two answers match after normalization."""
    if predicted is None or ground_truth is None:
        return False
    return normalize_answer(predicted) == normalize_answer(ground_truth)
