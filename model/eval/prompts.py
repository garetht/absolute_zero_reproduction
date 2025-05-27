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
    """
    Creates a formatted prompt string based on the problem configuration and which variable needs to be solved.

    This function generates an appropriate prompt by selecting between two predefined templates
    depending on whether the unknown variable is 'x' or 'y'. The prompt is formatted with the
    known values from the problem instance.

    :param problem: The problem instance containing the variable values and indicating which
                    variable is unknown
    :return: A formatted prompt string ready for use
    :rtype: str
    """
    if problem.blank == 'x':
        prompt = X_PROMPT.format(y=problem.y, prime=problem.prime)
    else:  # problem.blank == 'y'
        prompt = Y_PROMPT.format(x=problem.x, prime=problem.prime)
    return prompt
