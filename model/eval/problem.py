from dataclasses import dataclass, field

from typing import Optional
from typing_extensions import Literal

from custom_types import PrimeSample, TaskType


@dataclass
class Problem:
    prime: int
    x: Optional[int]
    y: Optional[int]
    blank: Literal['x', 'y', 'p']
    # For reproducible display
    desc: str = field(default='')

    def __repr__(self) -> str:
        """Return a nicely formatted string representation of the problem."""
        if self.blank == 'x':
            return f"Find x such that x * {self.y} ≡ 1 (mod {self.prime})"
        elif self.blank == 'y':
            return f"Find y such that {self.x} * y ≡ 1 (mod {self.prime})"
        else:
            return f"Find a p such that {self.x} * y ≡ 1 (mod {self.prime})"


    def to_prime_sample(self) -> PrimeSample:
        """Convert this Problem instance to a PrimeSample."""
        return PrimeSample(
            snippet=str(self.prime),
            function_io=[{
                'input_str': self.x if self.x is not None else '',
                'output_str': self.y if self.y is not None else ''
            }]
        )

    @staticmethod
    def from_prime_sample(prime_sample: PrimeSample, task_type: TaskType) -> 'Problem':
        match task_type:
            case TaskType.ABDUCTION:
                blank = 'x'
            case TaskType.DEDUCTION:
                blank = 'y'
            case TaskType.INDUCTION:
                blank = 'p'

        return Problem(
            prime=prime_sample.prime,
            x=prime_sample.function_io[0].input_str,
            y=prime_sample.function_io[0].output_str,
            blank=blank
        )
