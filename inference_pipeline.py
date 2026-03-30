"""
Advanced Inference Pipeline for NVIDIA Nemotron Reasoning Challenge
====================================================================
Implements test-time techniques to maximize accuracy:
1. Self-Consistency (Majority Voting, K=16)
2. Format Enforcement (regex cascade + retry)
3. Prompt Ensembling (3 system prompt variants)
4. Temperature Curriculum (easy→hard adaptive)
5. Answer Normalization

Designed for the Kaggle evaluation environment with vLLM.

This can be used standalone or integrated into the Kaggle submission notebook.
"""

import re
import sys
import os
from collections import Counter
from typing import List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_templates import (
    extract_boxed_answer,
    normalize_answer,
    detect_category,
    format_inference_prompt,
)


# ============================================================================
# PROMPT ENSEMBLING — 3 system prompt variants
# ============================================================================

SYSTEM_PROMPTS = {
    "analytical": (
        "You are a precise pattern recognition engine. "
        "Analyze the input-output examples carefully to identify the hidden transformation rule. "
        "State the rule explicitly, then apply it step-by-step to the test input. "
        "Your final answer MUST be in \\boxed{answer} format."
    ),
    "mathematical": (
        "You are an expert mathematician. "
        "Given a set of input-output examples, reverse-engineer the mathematical operation or formula. "
        "Show all computations clearly. "
        "Your final answer MUST be in \\boxed{answer} format."
    ),
    "methodical": (
        "You are a methodical problem solver. Follow this exact process:\n"
        "Step 1: List all input-output pairs\n"
        "Step 2: For each pair, hypothesize what operation maps input to output\n"
        "Step 3: Find the common rule\n"
        "Step 4: Verify the rule on ALL examples\n"
        "Step 5: Apply the verified rule to the test input\n"
        "Your final answer MUST be in \\boxed{answer} format."
    ),
}


# ============================================================================
# ANSWER EXTRACTION — robust cascade
# ============================================================================

def extract_answer_robust(text: str) -> Optional[str]:
    """
    Multi-strategy answer extraction:
    1. Standard \\boxed{} extraction
    2. Handle nested braces
    3. Look for "answer is: X" patterns
    4. Last line fallback
    """
    # Strategy 1: Standard boxed
    ans = extract_boxed_answer(text)
    if ans:
        return ans
    
    # Strategy 2: Alternative boxed formats
    alt_patterns = [
        r'\\boxed\s*\{(.+?)\}',
        r'\\text\{boxed\}\{(.+?)\}',
        r'\$\\boxed\{(.+?)\}\$',
        r'boxed\{(.+?)\}',
    ]
    for pat in alt_patterns:
        m = re.findall(pat, text)
        if m:
            return m[-1].strip()
    
    # Strategy 3: "answer is" patterns
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)',
        r'(?:result|output)\s+(?:is|=)[:\s]+(.+?)(?:\.|$)',
        r'(?:therefore|thus|hence|so)[,:\s]+(.+?)(?:\.|$)',
    ]
    for pat in answer_patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            candidate = m[-1].strip()
            # Clean up common artifacts
            candidate = re.sub(r'[\\$]', '', candidate).strip()
            if len(candidate) < 100:  # Sanity check
                return candidate
    
    return None


# ============================================================================
# SELF-CONSISTENCY — majority voting
# ============================================================================

def majority_vote(answers: List[str]) -> Optional[str]:
    """
    Perform majority voting on a list of extracted answers.
    Returns the most common answer after normalization.
    """
    if not answers:
        return None
    
    # Normalize all answers
    normalized = [normalize_answer(a) for a in answers if a is not None]
    normalized = [a for a in normalized if a]  # Remove empties
    
    if not normalized:
        return None
    
    counter = Counter(normalized)
    winner, count = counter.most_common(1)[0]
    
    # Return the original (un-normalized) version of the winning answer
    for orig, norm in zip(answers, [normalize_answer(a) for a in answers if a]):
        if norm == winner:
            return orig
    
    return winner


def confidence_score(answers: List[str]) -> float:
    """
    Compute confidence as the fraction of answers agreeing with the majority.
    """
    if not answers:
        return 0.0
    
    normalized = [normalize_answer(a) for a in answers if a is not None]
    normalized = [a for a in normalized if a]
    
    if not normalized:
        return 0.0
    
    counter = Counter(normalized)
    _, top_count = counter.most_common(1)[0]
    return top_count / len(normalized)


# ============================================================================
# INFERENCE ENGINE — orchestrates everything
# ============================================================================

class NemotronInferenceEngine:
    """
    Full inference pipeline with self-consistency + prompt ensembling.
    
    Usage with vLLM:
        engine = NemotronInferenceEngine(llm=vllm_llm, tokenizer=tokenizer)
        answer = engine.solve(prompt_text)
    """
    
    def __init__(
        self,
        llm=None,
        tokenizer=None,
        k_samples: int = 16,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        use_ensembling: bool = True,
        use_temperature_curriculum: bool = True,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.k_samples = k_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_ensembling = use_ensembling
        self.use_temperature_curriculum = use_temperature_curriculum
    
    def _build_prompt(self, task_prompt: str, system_prompt: str) -> str:
        """Build a full prompt string for the model."""
        inference_prompt = format_inference_prompt(task_prompt)
        
        # Try to use chat template if available
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Solve this reasoning task step-by-step.\nTask: {task_prompt}"},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return text
            except Exception:
                pass
        
        # Fallback: plain format (same as training)
        return inference_prompt
    
    def _generate_vllm(self, prompts: list, n: int, temperature: float):
        """Generate using vLLM."""
        from vllm import SamplingParams
        
        params = SamplingParams(
            n=n,
            temperature=max(temperature, 0.01),  # vLLM needs > 0
            top_p=0.95,
            max_tokens=self.max_tokens,
        )
        
        outputs = self.llm.generate(prompts, params)
        results = []
        for output in outputs:
            samples = [o.text for o in output.outputs]
            results.append(samples)
        return results
    
    def solve(self, task_prompt: str) -> dict:
        """
        Solve a single problem using the full inference pipeline.
        Returns dict with 'answer', 'confidence', 'all_answers', 'method'.
        """
        all_answers = []
        
        # Determine which system prompts to use
        if self.use_ensembling:
            prompts_to_use = list(SYSTEM_PROMPTS.items())
        else:
            prompts_to_use = [("analytical", SYSTEM_PROMPTS["analytical"])]
        
        # Temperature curriculum: try deterministic first
        if self.use_temperature_curriculum:
            temps = [0.1, self.temperature]
        else:
            temps = [self.temperature]
        
        for temp in temps:
            # Samples per prompt variant
            k_per_variant = max(1, self.k_samples // len(prompts_to_use))
            
            prompt_texts = []
            for variant_name, sys_prompt in prompts_to_use:
                prompt_text = self._build_prompt(task_prompt, sys_prompt)
                prompt_texts.append(prompt_text)
            
            # Generate all at once
            batch_results = self._generate_vllm(prompt_texts, k_per_variant, temp)
            
            for samples in batch_results:
                for sample in samples:
                    ans = extract_answer_robust(sample)
                    if ans:
                        all_answers.append(ans)
            
            # Temperature curriculum: if high confidence at low temp, stop early
            if temp == 0.1 and len(temps) > 1:
                conf = confidence_score(all_answers)
                if conf >= 0.8 and len(all_answers) >= 3:
                    break  # High confidence at low temp → skip higher temp
        
        # Majority vote
        final_answer = majority_vote(all_answers)
        conf = confidence_score(all_answers)
        
        # Format enforcement: if no answer found, try harder
        if final_answer is None:
            # Retry with a very explicit "just answer" prompt
            retry_prompt = self._build_prompt(
                task_prompt + "\n\nIMPORTANT: You MUST put your final answer inside \\boxed{}.",
                SYSTEM_PROMPTS["methodical"]
            )
            retry_results = self._generate_vllm([retry_prompt], 4, 0.3)
            for samples in retry_results:
                for sample in samples:
                    ans = extract_answer_robust(sample)
                    if ans:
                        all_answers.append(ans)
            
            final_answer = majority_vote(all_answers)
            conf = confidence_score(all_answers)
        
        # Ultimate fallback
        if final_answer is None:
            final_answer = "0"
            conf = 0.0
        
        return {
            "answer": final_answer,
            "confidence": conf,
            "num_answers": len(all_answers),
            "all_answers": all_answers[:10],  # Keep first 10 for debugging
        }
    
    def solve_batch(self, prompts: list) -> list:
        """Solve multiple problems. Returns list of result dicts."""
        results = []
        for i, prompt in enumerate(prompts):
            result = self.solve(prompt)
            if (i + 1) % 10 == 0:
                print(f"  Solved {i+1}/{len(prompts)} (conf={result['confidence']:.2f})")
            results.append(result)
        return results


# ============================================================================
# KAGGLE SUBMISSION HELPER
# ============================================================================

def create_submission(
    test_csv_path: str,
    llm,
    tokenizer,
    output_csv: str = "submission.csv",
    k_samples: int = 16,
    temperature: float = 0.6,
    use_ensembling: bool = True,
):
    """
    End-to-end: load test.csv → run inference → write submission.csv.
    
    Usage in Kaggle notebook:
        from inference_pipeline import create_submission
        create_submission(
            test_csv_path="/kaggle/input/.../test.csv",
            llm=vllm_engine,
            tokenizer=tokenizer,
            output_csv="/kaggle/working/submission.csv",
        )
    """
    import pandas as pd
    
    test_df = pd.read_csv(test_csv_path)
    
    # Auto-detect prompt column
    q_candidates = ['question', 'problem', 'prompt', 'instruction', 'text']
    cols = test_df.columns.tolist()
    q_col = next((c for c in cols if c.lower() in q_candidates), cols[-1])
    
    print(f"📋 Loaded {len(test_df)} test problems")
    
    engine = NemotronInferenceEngine(
        llm=llm,
        tokenizer=tokenizer,
        k_samples=k_samples,
        temperature=temperature,
        use_ensembling=use_ensembling,
    )
    
    answers = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        result = engine.solve(row[q_col])
        answers.append(result["answer"])
        confidences.append(result["confidence"])
        
        print(f"  [{idx+1}/{len(test_df)}] conf={result['confidence']:.2f} | answer={result['answer'][:50]}")
    
    # Build submission
    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "answer": answers,
    })
    submission_df.to_csv(output_csv, index=False)
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    print(f"\n✅ Submission saved: {output_csv}")
    print(f"   Avg confidence: {avg_conf:.2f}")
    print(f"   Total problems: {len(test_df)}")
    
    return submission_df
