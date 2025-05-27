from model.eval.prime_inversion import Problem

X_PROMPT = """<|im_start|>user
Given a prime number p and an integer y, find x such that:

x * {y} ≡ 1 (mod {prime})

Provide your answer as a single boxed number e.g. \[
\\boxed{{x}}
\]<|im_end|>
<|im_start|>assistant
"""
Y_PROMPT = """<|im_start|>user
Given a prime number p and an integer x, find y such that:

{x} * y ≡ 1 (mod {prime})

Provide your answer as a single boxed number within e.g. \[
\\boxed{{x}}
\]<|im_end|>
<|im_start|>assistant
"""


def create_prompt(problem: Problem) -> str:
    """Create a prompt for the given problem."""
    if problem.blank == 'x':
        prompt = X_PROMPT.format(y=problem.y, prime=problem.prime)
    else:  # problem.blank == 'y'
        prompt = Y_PROMPT.format(x=problem.x, prime=problem.prime)
    return prompt
