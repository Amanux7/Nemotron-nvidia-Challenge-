CRITIQUE_PROMPT = """You are an expert reviewer checking mathematical and logical reasoning.
Read the problem and the proposed solution. Identify any logical errors, miscalculations, or structural flaws.
If it is entirely correct, respond prominently with "CORRECT" on the first line.
If it is flawed, explain EXACTLY why step-by-step.

Problem:
{problem}

Proposed Solution:
{solution}

Critique:
"""

def generate_critique_prompt(problem: str, solution: str) -> str:
    return CRITIQUE_PROMPT.format(problem=problem, solution=solution)

def is_critique_correct(critique_response: str) -> bool:
    """
    A simple heuristic to check if the critic model thought the answer was correct.
    Looks for the word 'CORRECT' early in the response.
    """
    return "CORRECT" in critique_response[:100].upper()
