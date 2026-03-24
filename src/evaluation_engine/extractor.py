import re

def extract_boxed_answer(text: str) -> str:
    """
    Extracts the final answer from a model output. 
    Prioritizes extracting the contents of the last \boxed{} tag.
    Falls back to extracting the last number found in the text.
    """
    # Attempt to extract using a simple regex for \boxed{}
    # This regex handles single-level braces well, but might fail on heavily nested latex.
    matches = re.findall(r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}', text)
    if matches:
        return matches[-1].strip()
    
    # More robust fallback for nested braces: Find the last occurrence of \boxed{ 
    # and extract until the matching closing brace.
    last_boxed_idx = text.rfind(r'\boxed{')
    if last_boxed_idx != -1:
        start_idx = last_boxed_idx + 7
        brace_count = 1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i].strip()
    
    # Fallback 2: Find the last numerical value in the text
    numbers = re.findall(r'-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)', text)
    if numbers:
        return numbers[-1]
        
    # Return empty if nothing could be extracted
    return ""

if __name__ == "__main__":
    # Simple tests
    test_1 = "The answer is \\boxed{42}."
    test_2 = "The answer is \\boxed{\\frac{1}{2}}."
    test_3 = "My logic says the final result is 3.14"
    
    assert extract_boxed_answer(test_1) == "42"
    assert extract_boxed_answer(test_2) == "\\frac{1}{2}"
    assert extract_boxed_answer(test_3) == "3.14"
    print("Extractor tests passed!")
