from custom_types import BaseSample, TaskType, IOPair, Answer


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


def validate_formatting_and_correctness(response: str, task_type: TaskType) -> Answer:
    return validate_formatting_and_correctness_bulk([response], task_type)[0]


def validate_formatting_and_correctness_bulk(responses: list[str], task_type: TaskType) -> list[Answer]:
    pass # call validate by executing?


def create_sample_from_answer(answer: Answer, task_type: TaskType) -> BaseSample:
    pass
