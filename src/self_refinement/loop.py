import os
import sys

# Ensure src is in path so we can import modules logically from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_engine.templates import BASELINE_COT_PROMPT
from evaluation_engine.extractor import extract_boxed_answer
from self_refinement.critique import generate_critique_prompt, is_critique_correct
from self_refinement.improve import generate_improve_prompt

def run_refinement_logic(problem: str, model_generate_fn, max_iters: int = 2):
    """
    Simulates the core self-refinement loop for a single problem.
    model_generate_fn: A function that takes a prompt string and returns a generated string.
    """
    # Step 1: Initial generation
    initial_prompt = BASELINE_COT_PROMPT.format(problem=problem)
    solution = model_generate_fn(initial_prompt)
    
    current_solution = solution
    history = [{"iteration": 0, "solution": current_solution, "critique": None}]
    
    for i in range(max_iters):
        # Step 2: Critique
        critique_prompt = generate_critique_prompt(problem, current_solution)
        critique = model_generate_fn(critique_prompt)
        
        if is_critique_correct(critique):
            # The model believes its solution is correct, terminate loop early
            history.append({
                "iteration": i+1, 
                "status": "Terminated early (Correct)", 
                "solution": current_solution,
                "critique": critique
            })
            break
            
        # Step 3: Improve based on critique
        improve_prompt = generate_improve_prompt(problem, current_solution, critique)
        current_solution = model_generate_fn(improve_prompt)
        
        history.append({
            "iteration": i+1, 
            "solution": current_solution, 
            "critique": critique
        })
        
    final_answer = extract_boxed_answer(current_solution)
    return {
        "final_solution": current_solution,
        "final_answer": final_answer,
        "history": history
    }

if __name__ == "__main__":
    # Mocking a generative model for testing the loop logic locally
    def mock_model_generate(prompt):
        if "Critique:" in prompt:
            return "You forgot to carry the 1."
        elif "Revised" in prompt:
            return "Ah, I see. My initial math was wrong. The answer is \\boxed{42}."
        else:
            return "Let me think step-by-step. The answer is \\boxed{41}."
            
    print("Running mock self-refinement loop test...")
    res = run_refinement_logic("What is the ultimate answer to life, universe and everything?", mock_model_generate, max_iters=1)
    
    print("\n--- Mock Refinement Output ---")
    print(f"Initial extracted answer: {extract_boxed_answer(res['history'][0]['solution'])}")
    print(f"Critique received: {res['history'][1]['critique']}")
    print(f"Final extracted Refined answer: {res['final_answer']}")
