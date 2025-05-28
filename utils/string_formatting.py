import re
from typing import Optional
from dataclasses import dataclass
from custom_types import BaseSample, TaskType, Answer, PrimeSample, Problem, Role
from model.eval.prime_inversion import solve_modular_inverse
END_OF_USER_MESSAGE = """
<|im_end|>
<|im_start|>assistant
"""

BOXED_NUMBER = """
Provide your answer as a single boxed number within e.g. \[
\\boxed{{x}}
\]
"""

BOXED_NUMBER_X = """
Provide your answer as a single boxed number within e.g. \[
\\boxed{{x}}
\]
"""

BOXED_NUMBER_Y = """
Provide your answer as a single boxed number within e.g. \[
\\boxed{{y}}
\]
"""

BOXED_NUMBER_P = """
Provide your answer as a single boxed number within e.g. \[
\\boxed{{p}}
\]
"""
# TODO wrap this in a box?
BOXED_XY_PAIRS = """
Provide your answer as {num_io_pairs} equation(s) in the format:
equation_1
equation_2
...

Each equation should follow the pattern: x * y ≡ 1 (mod p)
"""

ABDUCTION_SOLVER_PROMPT = """<|im_start|>user
Given a prime number p and an integer y, find x such that:

x * {y} ≡ 1 (mod {prime})

{boxed_number}{end_message}"""

DEDUCTION_SOLVER_PROMPT = """<|im_start|>user
Given a prime number p and an integer x, find y such that:

{x} * y ≡ 1 (mod {prime})

{boxed_number}{end_message}"""

INDUCTION_SOLVER_PROMPT = """<|im_start|>user
Given integers x and y, find a p such that:

{x} * {y} ≡ 1 (mod p)

{boxed_number}{end_message}"""

ABDUCTION_PROPOSER_PROMPT = """<|im_start|>user
Task: Create a prime inversion equation with one missing input, x. There should be a single solution for x.
Using the reference example provided below, create your own unique equation.

Reference Example:
x * {y} ≡ 1 (mod {prime})

Create a new equation following this pattern:
x * [your_y] ≡ 1 (mod [your_prime])

{boxed_number}{end_message}"""

DEDUCTION_PROPOSER_PROMPT = """<|im_start|>user
Task: Create a prime inversion equation with one missing input, y. There should be a single solution for y.
Using the reference example provided below, create your own unique equation.

Reference Example:
{x} * y ≡ 1 (mod {prime})

Create a new equation following this pattern:
[your_x] * y ≡ 1 (mod [your_prime])

{boxed_number}{end_message}"""

INDUCTION_PROPOSER_PROMPT = """<|im_start|>user
Task: Create {num_io_pairs} prime inversion equation(s) with one missing prime, p. There should be at least one valid solution for p that works for all equations.
Using the reference example provided below, create your own unique equation(s).

Reference Example:
{x} * {y} ≡ 1 (mod p)

{boxed_number}{end_message}"""


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
            create_proposer_prompt(Problem.from_prime_sample(sample, task_type), num_io_pairs=num_io_pairs)
        case Role.SOLVER:
            create_solver_prompt(Problem.from_prime_sample(sample, task_type))


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
    for response in parsed_responses:
        # 1. Check types
        if not check_types(response, config['expect_types']):
            answers.append(INVALID_FORMATTING)
            continue

        # 2. Logic/solution check
        if not config['logic'](response):
            answers.append(INVALID_FORMATTING)
            continue

        answers.append(config['make_answer'](response))
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
            prompt = ABDUCTION_PROPOSER_PROMPT.format(y=problem.y, prime=problem.prime, 
                                                    boxed_number=BOXED_NUMBER_X,
                                                    end_message=END_OF_USER_MESSAGE)
        case TaskType.DEDUCTION:
            prompt = DEDUCTION_PROPOSER_PROMPT.format(x=problem.x, prime=problem.prime, 
                                                    boxed_number=BOXED_NUMBER_Y,
                                                    end_message=END_OF_USER_MESSAGE)
        case TaskType.INDUCTION:
            # Default to 1 if num_io_pairs is not specified
            pairs_count = num_io_pairs if num_io_pairs is not None else 1
            boxed_pairs = BOXED_XY_PAIRS.format(num_io_pairs=pairs_count)
            prompt = INDUCTION_PROPOSER_PROMPT.format(x=problem.x, y=problem.y, 
                                                    num_io_pairs=pairs_count,
                                                    boxed_number=boxed_pairs,
                                                    end_message=END_OF_USER_MESSAGE)
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
            prompt = ABDUCTION_SOLVER_PROMPT.format(y=problem.y, prime=problem.prime, 
                                                  boxed_number=BOXED_NUMBER_X,
                                                  end_message=END_OF_USER_MESSAGE)
        case "y":
            prompt = DEDUCTION_SOLVER_PROMPT.format(x=problem.x, prime=problem.prime, 
                                                  boxed_number=BOXED_NUMBER_Y,
                                                  end_message=END_OF_USER_MESSAGE)
        case "p":
            prompt = INDUCTION_SOLVER_PROMPT.format(x=problem.x, y=problem.y, 
                                                  boxed_number=BOXED_NUMBER_P,
                                                  end_message=END_OF_USER_MESSAGE)
        case _:
            raise ValueError(f"invalid blank value {problem.blank}")

    return prompt
