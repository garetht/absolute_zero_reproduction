import re

from typing import Optional, Callable
from dataclasses import dataclass
from custom_types import BaseSample, TaskType, IOPair, Answer
from model.eval.prime_inversion import validate_modular_inverse, solve_modular_inverse

BOXED_NUMBER = """
Provide your answer as a single boxed number within e.g. \[
\\boxed{{x}}
\]<|im_end|>
<|im_start|>assistant
"""

ABDUCTION_PROMPT = """<|im_start|>user
Given a prime number p and an integer y, find x such that:

x * {y} ≡ 1 (mod {prime})

{boxed_number}
"""

DEDUCTION_PROMPT = """<|im_start|>user
Given a prime number p and an integer x, find y such that:

{x} * y ≡ 1 (mod {prime})

{boxed_number}
"""

INDUCTION_PROMPT = """<|im_start|>user
Given integers x and y, find a p such that:

{x} * {y} ≡ 1 (mod p)

{boxed_number}
"""


@dataclass
class ModularEquation:
    x: int | str
    y: int | str
    p: int | str


def extract_modular_equation(text) -> Optional[ModularEquation]:
    """
    Extract x, y, and p from modular equation: x * y ≡ 1 (mod p)
    Returns a dictionary with the extracted values
    """
    # Simplified regex pattern
    pattern = r'(\w+) \* (\w+) ≡ 1 \(mod (\w+)\)'

    re_match = re.search(pattern, text)

    if re_match is not None:
        def try_parse_int(value_str):
            try:
                return int(value_str)
            except ValueError:
                return value_str

        result = ModularEquation(
            x=try_parse_int(re_match.group(1)),
            y=try_parse_int(re_match.group(2)),
            p=try_parse_int(re_match.group(3))
        )
        return result
    else:
        return ModularEquation(
            x='',
            y='',
            p=''
        )


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


def format_for_induction(program: BaseSample, num_io_pairs: int) -> str:
    pass


def format_for_abduction(program: BaseSample) -> str:
    pass


def format_for_deduction(program: BaseSample) -> str:
    pass


def format_task_prompts(sample: list[BaseSample], task_type: TaskType) -> list[str]:
    pass


def format_sample_from_io_pairs(valid_pairs_and_rewards: list[IOPair]) -> BaseSample:
    pass


def extract_io_pairs_from_string(response: str, num_io_pairs: int) -> list[IOPair]:
    pass


def validate_proposer_formatting_and_correctness(response: str, task_type: TaskType) -> Answer:
    return validate_formatting_and_correctness_bulk([response], task_type)[0]


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


def validate_formatting_and_correctness_bulk(
        responses: list[str],
        task_type: TaskType
) -> list[Answer]:
    answers = []

    # Define validation functions per task for generalization:
    check_map = {
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

    config = check_map[task_type]

    for response in responses:
        parsed_response = extract_modular_equation(response)
        # 1. Check types
        if not check_types(parsed_response, config['expect_types']):
            answers.append(INVALID_FORMATTING)
            continue
        # 2. Logic/solution check
        if not config['logic'](parsed_response):
            answers.append(INCORRECT_ANSWER)
            continue
        # 3. Success
        answers.append(config['make_answer'](parsed_response))

    return answers


def create_sample_from_answer(answer: Answer, task_type: TaskType) -> BaseSample:
    pass
