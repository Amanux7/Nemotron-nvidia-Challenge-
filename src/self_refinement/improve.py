IMPROVE_PROMPT = """You are an expert mathematical and logical reasoner.
You previously attempted a problem but made a mistake.
Using the provided critique, rewrite the step-by-step reasoning to fix the error.
After your revised reasoning, you MUST provide the final answer enclosed in a \\boxed{{}} tag.

Problem:
{problem}

Previous Flawed Solution:
{solution}

Critique of flaw:
{critique}

Revised Step-by-step reasoning:
"""

def generate_improve_prompt(problem: str, solution: str, critique: str) -> str:
    return IMPROVE_PROMPT.format(problem=problem, solution=solution, critique=critique)
