import random
from dataclasses import field, dataclass

from typing import Optional
from typing_extensions import Literal

from custom_types import PrimeSample


def solve_modular_inverse(p: int, x=None, y=None, verbose: bool = False):
    """
    Solves for the unknown in xy ≡ 1 mod p, where p is prime, and one of x or y is given.

    Args:
        p (int): A prime modulus.
        x (int, optional): The value of x. If None, y must be given.
        y (int, optional): The value of y. If None, x must be given.
        verbose (bool, optional): Whether to print solving steps.

    Returns:
        int: The value of the unknown variable (mod p).
    """
    if (x is None and y is None) or (x is not None and y is not None):
        raise ValueError("Exactly one of x or y must be provided.")
    if x is not None:
        # Compute modular inverse of x mod p
        result = pow(x, -1, p)
        if verbose:
            print(f"Given x = {x}, solving for y such that x*y ≡ 1 mod {p}: y = {result}")
        return result
    else:
        # Compute modular inverse of y mod p
        result = pow(y, -1, p)
        if verbose:
            print(f"Given y = {y}, solving for x such that x*y ≡ 1 mod {p}: x = {result}")
        return result


@dataclass
class Problem:
    prime: int
    x: Optional[int]
    y: Optional[int]
    blank: Literal['x', 'y']
    # For reproducible display
    desc: str = field(default='')

    def __repr__(self) -> str:
        """Return a nicely formatted string representation of the problem."""
        if self.blank == 'x':
            return f"Find x such that x * {self.y} ≡ 1 (mod {self.prime})"
        else:
            return f"Find y such that {self.x} * y ≡ 1 (mod {self.prime})"


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
    def from_prime_sample(prime_sample: PrimeSample, blank: Literal['x', 'y']) -> 'Problem':
        return Problem(
            prime=prime_sample.prime,
            x=prime_sample.function_io[0].input_str,
            y=prime_sample.function_io[0].output_str,
            blank=blank
        )


def modular_inverse(a: int, p: int) -> int:
    # Using Fermat's little theorem to compute modular inverse: a^(p-2) mod p
    return pow(a, p - 2, p)


def generate_problems(n: int, primes: list[int], seed: int = 42) -> list[Problem]:
    """
    Generate a list of modular inverse problems for evaluation.

    This function creates n problems where each problem involves finding either x or y
    in the equation xy ≡ 1 (mod p), where p is a prime number. For each problem,
    one variable is randomly chosen to be the unknown (blank), and the other is given.

    Args:
        n (int): The number of problems to generate.
        primes (list[int]): A list of prime numbers to choose from as moduli.
        seed (int, optional): Random seed for reproducible problem generation. Defaults to 42.

    Returns:
        list[Problem]: A list of Problem instances, each containing:
            - prime: The prime modulus
            - x: The x value (or None if x is the unknown)
            - y: The y value (or None if y is the unknown)
            - blank: Either 'x' or 'y' indicating which variable to solve for
            - desc: A human-readable description of the problem

    Example:
        >>> primes = [7, 11, 13]
        >>> problems = generate_problems(2, primes, seed=123)
        >>> len(problems)
        2
        >>> problems[0].prime in primes
        True
    """
    r = random.Random(seed)
    problems = []
    for _ in range(n):
        p = r.choice(primes)
        x = r.randint(1, p - 1)
        y = modular_inverse(x, p)
        blank = r.choice(['x', 'y'])
        if blank == 'x':
            prob = Problem(prime=p, x=x, y=y, blank='x', desc=f"Find x given y={y}, p={p}")
        else:
            prob = Problem(prime=p, x=x, y=y, blank='y', desc=f"Find y given x={x}, p={p}")
        problems.append(prob)
    return problems
