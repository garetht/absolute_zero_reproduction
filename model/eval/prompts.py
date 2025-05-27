from custom_types import Problem

BOXED_NUMBER = """
Provide your answer as a single boxed number within e.g. \[
\\boxed{{x}}
\]<|im_end|>
<|im_start|>assistant
"""

X_PROMPT = """<|im_start|>user
Given a prime number p and an integer y, find x such that:

x * {y} ≡ 1 (mod {prime})

{boxed_number}
"""
Y_PROMPT = """<|im_start|>user
Given a prime number p and an integer x, find y such that:

{x} * y ≡ 1 (mod {prime})

{boxed_number}
"""

P_PROMPT = """<|im_start|>user
Given integers x and y, find a p such that:

{x} * {y} ≡ 1 (mod p)

{boxed_number}
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
    match problem.blank:
        case "x":
            prompt = X_PROMPT.format(y=problem.y, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case "y":
            prompt = Y_PROMPT.format(x=problem.x, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case "p":
            prompt = P_PROMPT.format(x=problem.x, y=problem.y, boxed_number=BOXED_NUMBER)
        case _:
            raise ValueError(f"invalid blank value {problem.blank}")

    return prompt
