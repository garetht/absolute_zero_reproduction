from typing import Optional

from custom_types import Reward, Answer, IOPair

"""
Validates that the provided problem is correct python syntax, and that the input generates a determinstic answer. Outputs a dict containing the score for formatting and the score for correctness.
"""
def validate_by_executing_induction(io_pairs: list[IOPair]) -> tuple[list[IOPair], Answer]:
    """
    Returns
        valid IO pairs
        rewards
    """
    pass
