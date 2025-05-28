from custom_types import Problem
from utils.string_formatting import BOXED_NUMBER, ABDUCTION_PROMPT, DEDUCTION_PROMPT, INDUCTION_PROMPT


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
            prompt = ABDUCTION_PROMPT.format(y=problem.y, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case "y":
            prompt = DEDUCTION_PROMPT.format(x=problem.x, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case "p":
            prompt = INDUCTION_PROMPT.format(x=problem.x, y=problem.y, boxed_number=BOXED_NUMBER)
        case _:
            raise ValueError(f"invalid blank value {problem.blank}")

    return prompt
