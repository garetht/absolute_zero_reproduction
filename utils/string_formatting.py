import re

from typing import Optional, Callable
from dataclasses import dataclass
from custom_types import BaseSample, TaskType, IOPair, Answer, PrimeSample, Problem, Role
from model.eval.prime_inversion import validate_modular_inverse, solve_modular_inverse

BOXED_NUMBER = """
Provide your answer as a single boxed number within e.g. \[
\\boxed{{x}}
\]<|im_end|>
<|im_start|>assistant
"""

ABDUCTION_SOLVER_PROMPT = """<|im_start|>user
Given a prime number p and an integer y, find x such that:

x * {y} ≡ 1 (mod {prime})

{boxed_number}
"""

DEDUCTION_SOLVER_PROMPT = """<|im_start|>user
Given a prime number p and an integer x, find y such that:

{x} * y ≡ 1 (mod {prime})

{boxed_number}
"""

INDUCTION_SOLVER_PROMPT = """<|im_start|>user
Given integers x and y, find a p such that:

{x} * {y} ≡ 1 (mod p)

{boxed_number}
"""

ABDUCTION_PROPOSER_PROMPT = """<|im_start|>user

"""

DEDUCTION_PROPOSER_PROMPT = """<|im_start|>user

"""

INDUCTION_PROPOSER_PROMPT = """<|im_start|>user

"""


@dataclass
class ModularEquation:
    x: int | str
    y: int | str
    p: int | str


def extract_modular_equations(text) -> list[ModularEquation]:
    """
    Extract x, y, and p from modular equation: x * y ≡ 1 (mod p)
    Returns a list of ModularEquation objects for all matches found
    """
    # Simplified regex pattern
    pattern = r'(\w+) \* (\w+) ≡ 1 \(mod (\w+)\)'

    re_matches = re.findall(pattern, text)

    def try_parse_int(value_str: str) -> int | str:
        try:
            return int(value_str)
        except ValueError:
            return value_str

    results = []
    for match in re_matches:
        result = ModularEquation(
            x=try_parse_int(match[0]),
            y=try_parse_int(match[1]),
            p=try_parse_int(match[2])
        )
        results.append(result)

    return results


def extract_boxed_number(text: str) -> Optional[int]:
    """
    Extract a number from LaTeX \\boxed{} notation in the given text.

    Args:
        text: The text to search for boxed numbers

    Returns:
        The integer found within \\boxed{} notation, or None if no match is found
    """
    # Regex pattern to match \boxed{<number>} and extract the number
    regexp_match = re.search(r"\\boxed\{([+-]?\d+)\}", text)
    if regexp_match:
        return int(regexp_match.group(1))
    else:
        return None


def format_as_string(sample: PrimeSample, task_type: TaskType, role: Role, num_io_pairs: Optional[int] = 0) -> str:
    match role:
        case Role.PROPOSER:
            return create_proposer_prompt(Problem.from_prime_sample(sample, task_type), num_io_pairs=num_io_pairs)
        case Role.SOLVER:
            return create_solver_prompt(Problem.from_prime_sample(sample, task_type))

    raise ValueError(f"unexpected role {role}")

def validate_proposer_formatting_and_correctness(response: str, task_type: TaskType) -> Answer:
    return validate_proposer_formatting_and_correctness_bulk([response], task_type)[0]


INVALID_FORMATTING = Answer(
    input=None,
    program=None,
    output=None,
    reward=-1.0
)

INCORRECT_ANSWER = Answer(
    input=None,
    program=None,
    output=None,
    reward=-0.5
)


def check_types(parsed: object, expect_types: dict[str, bool]) -> bool:
    for field, should_be_num in expect_types.items():
        val = getattr(parsed, field)
        if should_be_num:
            if not isinstance(val, int):
                return False
        else:
            if isinstance(val, int):
                return False
    return True


CHECK_MAP = {
    TaskType.ABDUCTION: {
        "expect_types": {"x": False, "y": True, "p": True},
        "logic": lambda parsed: len(solve_modular_inverse(y=parsed.y, p=parsed.p)) == 1,
        "make_answer": lambda parsed: Answer(input=None, program=parsed.p, output=parsed.y, reward=0.0)
    },
    TaskType.DEDUCTION: {
        "expect_types": {"y": False, "x": True, "p": True},
        "logic": lambda parsed: len(solve_modular_inverse(x=parsed.x, p=parsed.p)) == 1,
        "make_answer": lambda parsed: Answer(input=parsed.x, program=parsed.p, output=None, reward=0.0)
    },
    TaskType.INDUCTION: {
        "expect_types": {"p": False, "x": True, "y": True},
        "logic": lambda parsed: len(solve_modular_inverse(x=parsed.x, y=parsed.y)) > 0,
        "make_answer": lambda parsed: Answer(input=parsed.x, program=None, output=parsed.y, reward=0.0)
    }
}

def validate_proposer_formatting_and_correctness_bulk(
        responses: list[str],
        task_type: TaskType
) -> list[Answer]:
    """
    Validate multiple responses for a given task type.

    Args:
        responses: List of response strings to validate
        task_type: The type of task (ABDUCTION, DEDUCTION, or INDUCTION)

    Returns:
        List of Answer objects corresponding to each response
    """
    # Define validation functions per task for generalization:


    config = CHECK_MAP[task_type]

    # Process all responses using the extracted function
    return validate_responses(responses, config)


def validate_responses(responses: list[str], config: dict) -> list[Answer]:
    """
    Validate a list of responses using the provided configuration.

    Args:
        responses: List of response strings to validate
        config: Dictionary containing validation configuration with keys:
            - expect_types: Expected types for x, y, p
            - logic: Function to validate the logic/solution
            - make_answer: Function to create Answer object

    Returns:
        List of Answer objects
    """
    all_answers = []

    for response in responses:
        answers = validate_single_response(response, config)
        all_answers.extend(answers)

    return all_answers


def validate_single_response(response: str, config: dict) -> list[Answer]:
    """
    Validate a single response using the provided configuration.

    Args:
        response: Response string to validate
        config: Dictionary containing validation configuration

    Returns:
        Answer object (or INVALID_FORMATTING/INCORRECT_ANSWER on failure)
    """
    parsed_responses = extract_modular_equations(response)
    answers = []

    if len(parsed_responses) == 0:
        return [INVALID_FORMATTING]

    for response in parsed_responses:
        # 1. Check types
        if not check_types(response, config['expect_types']):
            answers.append(INVALID_FORMATTING)
            continue

        # 2. Logic/solution check
        if not config['logic'](response):
            answers.append(INVALID_FORMATTING)
            continue

        answer = config['make_answer'](response)
        solutions = solve_modular_inverse(p=answer.program, x=answer.input, y=answer.output)

        if len(solutions) == 0:
            answers.append(INCORRECT_ANSWER)
        else:
            answers.append(answer)

    # 3. Success
    return answers


def create_sample_from_answer(answer: Answer, task_type: TaskType) -> BaseSample:
    pass


def validate_solver_formatting_and_correctness(response: str, task_type: TaskType, sample: PrimeSample) -> Answer:
    parsed_number = extract_boxed_number(response)
    if parsed_number is None:
        return INVALID_FORMATTING

    is_correct = False
    match task_type:
        case TaskType.ABDUCTION:
            is_correct = parsed_number == sample.function_io[0].input_str
        case TaskType.DEDUCTION:
            is_correct = parsed_number == sample.function_io[0].output_str
        case TaskType.INDUCTION:
            is_correct = parsed_number == sample.prime

    if is_correct:
        return Answer(
            input=sample.function_io[0].input_str,
            output=sample.function_io[0].output_str,
            program=sample.prime,
            reward=1.0
        )
    else:
        return INCORRECT_ANSWER


def create_proposer_prompt(problem: Problem, num_io_pairs: Optional[int] = None) -> str:
    match problem.task_type:
        case TaskType.ABDUCTION:
            prompt = ABDUCTION_PROPOSER_PROMPT.format(y=problem.y, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case TaskType.DEDUCTION:
            prompt = DEDUCTION_PROPOSER_PROMPT.format(x=problem.x, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case TaskType.INDUCTION:
            prompt = INDUCTION_PROPOSER_PROMPT.format(x=problem.x, y=problem.y, boxed_number=BOXED_NUMBER,
                                                      num_io_pairs=num_io_pairs)
        case _:
            raise ValueError(f"invalid blank value {problem.blank}")

    return prompt


def create_solver_prompt(problem: Problem) -> str:
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
            prompt = ABDUCTION_SOLVER_PROMPT.format(y=problem.y, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case "y":
            prompt = DEDUCTION_SOLVER_PROMPT.format(x=problem.x, prime=problem.prime, boxed_number=BOXED_NUMBER)
        case "p":
            prompt = INDUCTION_SOLVER_PROMPT.format(x=problem.x, y=problem.y, boxed_number=BOXED_NUMBER)
        case _:
            raise ValueError(f"invalid blank value {problem.blank}")

    return prompt
