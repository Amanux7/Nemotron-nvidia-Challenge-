"""
Verify and Filter Generated CoT Data
======================================
Post-processing QA step: validates generated traces, removes bad samples,
and produces the final clean dataset for fine-tuning.

Usage:
    python verify_and_filter.py \
        --input_file train_cot.jsonl \
        --output_file train_cot_clean.jsonl
"""

import argparse
import json
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_templates import extract_boxed_answer, answers_match


def validate_trace(record: dict, min_tokens: int = 30, max_tokens: int = 3000) -> tuple:
    """
    Validate a single CoT trace. Returns (is_valid, reason).
    """
    trace = record.get("reasoning_trace", "")
    answer = record.get("answer", "")
    full_trace = record.get("full_trace", "")
    
    # 1. Must have a non-empty reasoning trace
    if not trace or len(trace.strip()) < 10:
        return False, "empty_or_too_short_reasoning"
    
    # 2. Length bounds (in characters as proxy for tokens)
    if len(trace) < min_tokens * 3:  # ~3 chars per token
        return False, "trace_too_short"
    if len(trace) > max_tokens * 5:
        return False, "trace_too_long"
    
    # 3. Must have answer
    if not answer or answer.strip() == "":
        return False, "missing_answer"
    
    # 4. Verify boxed answer in full trace matches ground truth
    if full_trace:
        extracted = extract_boxed_answer(full_trace)
        if extracted and not answers_match(extracted, answer):
            return False, "answer_mismatch"
    
    # 5. Check for hallucination signals
    hallucination_markers = [
        "I don't know",
        "I cannot determine",
        "it's impossible to",
        "there is no pattern",
        "I apologize",
        "As an AI",
    ]
    trace_lower = trace.lower()
    for marker in hallucination_markers:
        if marker.lower() in trace_lower:
            return False, f"hallucination_marker: {marker}"
    
    # 6. Check reasoning has actual analytical content (not just filler)
    # Must contain at least some numbers or logical connectors
    has_analysis = bool(re.search(r'\d+|step|rule|pattern|result|apply|convert|shift|xor|cipher', trace_lower))
    if not has_analysis:
        return False, "no_analytical_content"
    
    return True, "valid"


def main():
    parser = argparse.ArgumentParser(description="Verify and filter CoT data")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL from generate_cot_data.py")
    parser.add_argument("--output_file", type=str, default=None, help="Output clean JSONL")
    parser.add_argument("--min_tokens", type=int, default=30, help="Min reasoning length (token estimate)")
    parser.add_argument("--max_tokens", type=int, default=3000, help="Max reasoning length (token estimate)")
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = args.input_file.replace('.jsonl', '_clean.jsonl')
    
    # Load all records
    records = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"📄 Loaded {len(records)} raw CoT records")
    
    # Validate
    valid_records = []
    rejection_reasons = Counter()
    category_stats = Counter()
    
    for record in records:
        is_valid, reason = validate_trace(record, args.min_tokens, args.max_tokens)
        if is_valid:
            valid_records.append(record)
            category_stats[record.get('category', 'unknown')] += 1
        else:
            rejection_reasons[reason] += 1
    
    # Save clean data
    with open(args.output_file, 'w') as f:
        for record in valid_records:
            f.write(json.dumps(record) + '\n')
    
    # Print stats
    print(f"\n{'='*60}")
    print(f"📊 FILTERING RESULTS")
    print(f"   Input:  {len(records)}")
    print(f"   Output: {len(valid_records)} ({len(valid_records)/max(len(records),1)*100:.1f}%)")
    print(f"   Rejected: {len(records) - len(valid_records)}")
    
    if rejection_reasons:
        print(f"\n   Rejection reasons:")
        for reason, count in rejection_reasons.most_common():
            print(f"     {reason}: {count}")
    
    print(f"\n   Category distribution (valid):")
    for cat, count in category_stats.most_common():
        print(f"     {cat}: {count}")
    
    print(f"\n   📄 Clean output: {args.output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
