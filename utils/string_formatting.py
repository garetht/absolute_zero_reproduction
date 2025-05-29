import random
import re
from typing import Optional
from dataclasses import dataclass
from constants import MAXIMUM_PRIME
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
Provide your answer as a single boxed number e.g. 
\[
\\boxed{{x}}
\]
"""

BOXED_NUMBER_Y = """
Provide your answer as a single boxed number e.g. \[
\\boxed{{y}}
\]
"""

BOXED_NUMBER_P = """
Provide your answer as a single boxed number e.g. \[
\\boxed{{p}}
\]
"""

BOXED_XY_PAIRS = """
Provide your answer as {num_io_pairs} equation(s) each in a separate boxed equation e.g. \[
\\boxed{{equation_1}}
\], \[
\\boxed{{equation_2}}
\],
...
"""

BOXED_EQUATION = """
Answer with a single boxed equation in this format: \[
\\boxed{{x * y ≡ 1 (mod p)}}
\]
"""

ABDUCTION_SOLVER_PROMPT = """<|im_start|>user
## Task:
Given a prime number p and an integer y, find x such that:

x * {y} ≡ 1 (mod {prime})

### Formatting:{boxed_number}{end_message}"""

OLD_ABDUCTION_SOLVER_PROMPT = """<|im_start|>user
Given a prime number p and an integer y, find x such that:

x * {y} ≡ 1 (mod {prime})

{boxed_number}{end_message}"""

DEDUCTION_SOLVER_PROMPT = """<|im_start|>user
## Task:
Given a prime p and an integer x, solve for y:

{x} * y ≡ 1 (mod {prime})

### Formatting:{boxed_number}{end_message}"""

INDUCTION_SOLVER_PROMPT = """<|im_start|>user
## Task:
Given integers x and y, find a prime number p <= {maximum_prime} such that:

{x} * {y} ≡ 1 (mod p)

### Formatting:{boxed_number}{end_message}"""

ABDUCTION_PROPOSER_PROMPT = """<|im_start|>user
## Task:
Create a prime inversion equation x * y ≡ 1 (mod p) filling in y and p with positive integers and keeping x as is. 

Using the reference example provided below, create your own unique equation with a different value of y and p to improve your ability to invert numbers modulo a prime.

Reference Example:
x * {y} ≡ 1 (mod {prime})

### Evaluation Criteria:
- p must be a positive prime integer satisfying 5 <= p <= {maximum_prime}.
- p should be large enough such that you cannot find the inverse of r for all 1 <= r < p.
- p should be small enough such that you can find the inverse of some r in 1 <= r < p.
- y should satisfy 1 <= y < p.

### Formatting:{boxed_equation}{end_message}"""

DEDUCTION_PROPOSER_PROMPT = """<|im_start|>user
## Task:
Create a prime inversion equation x * y ≡ 1 (mod p) filling in x and p with positive integers and keeping y as is. 

Using the reference example provided below, create your own unique equation with a different value of x and p to improve your ability to invert numbers modulo a prime.

Reference Example:
{x} * y ≡ 1 (mod {prime})

### Evaluation Criteria:
- p must be a positive prime integer satisfying 5 <= p <= {maximum_prime}.
- p should be large enough such that you cannot find the inverse of r for all 1 <= r < p.
- p should be small enough such that you can find the inverse of some r in 1 <= r < p.
- x should satisfy 1 <= x < p.

### Formatting:{boxed_equation}{end_message}"""

INDUCTION_PROPOSER_PROMPT = """<|im_start|>user
## Task: 
Create {num_io_pairs} prime inversion equation(s) x * y ≡ 1 (mod p) filling in x and y while keeping the prime p as is.

Using the reference example provided below, create your own unique equations with different values of x and y --- **but the same p** --- to improve your ability to invert numbers modulo a prime.

Reference Example:
{induction_examples}

### Evaluation Criteria:
- You should have a positive prime integer p in mind when generating **all** the pairs.
- p should satisfying {num_io_pairs} < p <= {maximum_prime}.
- Make sure that the pair(s) of x and y are unique.
- Make sure that there are {num_io_pairs} pair(s) of x and y.
- **Do not include the prime p in your equations. Keep it as is.**
- If the letter p **DOES NOT** appear in your equation, you will be penalized.

### Formatting:{boxed_pairs}{end_message}"""


@dataclass
class ModularEquation:
    x: int | str
    y: int | str
    p: int | str


def extract_modular_equations(text) -> list[ModularEquation]:
    """
    Extract x, y, and p from LaTeX modular equation format: \[x \times y \equiv 1 \pmod{p} \]
    Returns a list of ModularEquation objects for all matches found
    """
    # LaTeX format regex pattern: \[37 \times 69 \equiv 1 \pmod{421} \]
    pattern = (
        r"\\?\[\s*(\w+)\s*\\times\s*(\w+)\s*\\equiv\s*1\s*\\pmod\{\s*(\w+)\s*\}\s*\\?\]"
    )

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
            p=try_parse_int(match[2]),
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


def format_as_string(
    sample: PrimeSample,
    task_type: TaskType,
    role: Role,
    num_io_pairs: Optional[int] = 0,
) -> str:
    match role:
        case Role.PROPOSER:
            return create_proposer_prompt(
                Problem.from_prime_sample(sample, task_type), num_io_pairs=num_io_pairs
            )
        case Role.SOLVER:
            return create_solver_prompt(Problem.from_prime_sample(sample, task_type))

    raise ValueError(f"unexpected role {role}")


def validate_proposer_formatting_and_correctness(
    response: str, task_type: TaskType
) -> Answer:
    return validate_proposer_formatting_and_correctness_bulk([response], task_type)[0]


INVALID_FORMATTING = Answer(input=None, program=None, output=None, reward=-1.0)

INCORRECT_ANSWER = Answer(input=None, program=None, output=None, reward=-0.5)


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
        "make_answer": lambda parsed: Answer(
            input=None, program=parsed.p, output=parsed.y, reward=0.0
        ),
    },
    TaskType.DEDUCTION: {
        "expect_types": {"y": False, "x": True, "p": True},
        "logic": lambda parsed: len(solve_modular_inverse(x=parsed.x, p=parsed.p)) == 1,
        "make_answer": lambda parsed: Answer(
            input=parsed.x, program=parsed.p, output=None, reward=0.0
        ),
    },
    TaskType.INDUCTION: {
        "expect_types": {"p": False, "x": True, "y": True},
        "logic": lambda parsed: len(solve_modular_inverse(x=parsed.x, y=parsed.y)) > 0,
        "make_answer": lambda parsed: Answer(
            input=parsed.x, program=None, output=parsed.y, reward=0.0
        ),
    },
}


def validate_proposer_formatting_and_correctness_bulk(
    responses: list[str], task_type: TaskType
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


def validate_single_response(str_response: str, config: dict) -> list[Answer]:
    """
    Validate a single response using the provided configuration.

    Args:
        response: Response string to validate
        config: Dictionary containing validation configuration

    Returns:
        Answer object (or INVALID_FORMATTING/INCORRECT_ANSWER on failure)
    """
    parsed_responses = extract_modular_equations(str_response)
    answers = []

    if len(parsed_responses) == 0:
        return [INVALID_FORMATTING]

    for response in parsed_responses:
        # 1. Check types
        if not check_types(response, config["expect_types"]):
            answers.append(INVALID_FORMATTING)
            continue

        # 2. Logic/solution check
        if not config["logic"](response):
            answers.append(INVALID_FORMATTING)
            continue

        answer = config["make_answer"](response)
        solutions = solve_modular_inverse(
            p=answer.program, x=answer.input, y=answer.output
        )

        if len(solutions) == 0:
            answers.append(INCORRECT_ANSWER)
        else:
            answers.append(answer)

    # 3. Success
    return answers

def create_problem_from_answer(answer: Answer, task_type: TaskType) -> Problem:
    """Create a Problem from an Answer object"""
    assert answer.program is not None, "Answer must have a program (prime) defined"    # Initialize x and y
    x = answer.input
    y = answer.output    # Compute missing values based on task type
    if task_type == TaskType.ABDUCTION:
        # For abduction, we have y and p, need to compute x
        if x is None and y is not None:
            solutions = solve_modular_inverse(y=y, p=answer.program)
            if solutions:
                x = solutions.pop()
    elif task_type == TaskType.DEDUCTION:
        # For deduction, we have x and p, need to compute y
        if y is None and x is not None:
            solutions = solve_modular_inverse(x=x, p=answer.program)
            if solutions:
                y = solutions.pop()
    elif task_type == TaskType.INDUCTION:
        # For induction, we should have both x and y already
        # But if we're missing the prime, we can't create a valid problem
        pass    

    return Problem(
        prime=answer.program,
        x_list=[x],
        y_list=[y],
        task_type=task_type
    )

# def create_problem_from_answer(answer: Answer, task_type: TaskType) -> Problem:
#     """Create a Problem from an Answer object"""
#     assert answer.program is not None, "Answer must have a program (prime) defined"
#     return Problem(
#         prime=answer.program,
#         x_list=[answer.input] if answer.input is not None else [],
#         y_list=[answer.output] if answer.output is not None else [], 
#         task_type=task_type
#     )

def validate_solver_formatting_and_correctness(response: str, task_type: TaskType, sample: Problem) -> Answer:
    parsed_number = extract_boxed_number(response)
    if parsed_number is None:
        return INVALID_FORMATTING

    is_correct = False
    match task_type:
        case TaskType.ABDUCTION:
            is_correct = parsed_number == sample.x
        case TaskType.DEDUCTION:
            is_correct = parsed_number == sample.y
        case TaskType.INDUCTION:
            is_correct = parsed_number == sample.prime

    if is_correct:
        return Answer(
            input=sample.x,
            output=sample.y,
            program=sample.prime,
            reward=1.0,
        )
    else:
        return INCORRECT_ANSWER


def create_proposer_prompt(problem: Problem, num_io_pairs: Optional[int] = None) -> str:
    def generate_induction_examples(num_pairs: int) -> str:
        """
        Generate a string of example equations for the induction task.
        """
        r = random.Random(42)  # Sorry, but hardcoding a seed here for simplicity
        p = 101  # Sorry, but hardcoding a prime here for simplicity
        examples = []
        for _ in range(num_pairs):
            x = r.randint(1, p - 1)
            y = pow(x, -1, p)  # Modular inverse of x mod p
            examples.append(f"{x} * {y} ≡ 1 (mod p)")
        return "\n".join(examples)

    match problem.task_type:
        case TaskType.ABDUCTION:
            prompt = ABDUCTION_PROPOSER_PROMPT.format(
                y=problem.y,
                prime=problem.prime,
                maximum_prime=MAXIMUM_PRIME,
                boxed_equation=BOXED_EQUATION,
                end_message=END_OF_USER_MESSAGE,
            )
        case TaskType.DEDUCTION:
            prompt = DEDUCTION_PROPOSER_PROMPT.format(
                x=problem.x,
                prime=problem.prime,
                maximum_prime=MAXIMUM_PRIME,
                boxed_equation=BOXED_EQUATION,
                end_message=END_OF_USER_MESSAGE,
            )
        case TaskType.INDUCTION:
            # Default to 1 if num_io_pairs is not specified
            pairs_count = num_io_pairs if num_io_pairs is not None else 1
            boxed_pairs = BOXED_XY_PAIRS.format(num_io_pairs=pairs_count)

            prompt = INDUCTION_PROPOSER_PROMPT.format(
                x=problem.x,
                y=problem.y,
                maximum_prime=MAXIMUM_PRIME,
                num_io_pairs=pairs_count,
                induction_examples=generate_induction_examples(pairs_count),
                boxed_pairs=boxed_pairs,
                end_message=END_OF_USER_MESSAGE,
            )
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
            prompt = ABDUCTION_SOLVER_PROMPT.format(
                y=problem.y,
                prime=problem.prime,
                boxed_number=BOXED_NUMBER_X,
                end_message=END_OF_USER_MESSAGE,
            )
        case "y":
            prompt = DEDUCTION_SOLVER_PROMPT.format(
                x=problem.x,
                prime=problem.prime,
                boxed_number=BOXED_NUMBER_Y,
                end_message=END_OF_USER_MESSAGE,
            )
        case "p":
            prompt = INDUCTION_SOLVER_PROMPT.format(
                x=problem.x,
                y=problem.y,
                maximum_prime=MAXIMUM_PRIME,
                boxed_number=BOXED_NUMBER_P,
                end_message=END_OF_USER_MESSAGE,
            )
        case _:
            raise ValueError(f"invalid blank value {problem.blank}")

    return prompt


from model.eval.prime_inversion import generate_problems

if __name__ == "__main__":
    task_type = TaskType.INDUCTION
    print("-------------------------")
    print(f"[TASK TYPE] {task_type}")
    print("-------------------------")
    problems = generate_problems(1, [5, 7, 11], seed=0)
    for problem in problems:
        problem.task_type = task_type
        solver_prompt = create_solver_prompt(problem)
        proposer_prompt = create_proposer_prompt(problem, num_io_pairs=5)
        print(f"[S] {solver_prompt}")
        print("-------------------------")
        print(f"[P] {proposer_prompt}")
        print("-------------------------")
