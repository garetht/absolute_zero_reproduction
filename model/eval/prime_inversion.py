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
    random.seed(seed)
    problems = []
    for _ in range(n):
        p = random.choice(primes)
        x = random.randint(1, p - 1)
        y = modular_inverse(x, p)
        blank = random.choice(['x', 'y'])
        if blank == 'x':
            prob = Problem(prime=p, x=x, y=y, blank='x', desc=f"Find x given y={y}, p={p}")
        else:
            prob = Problem(prime=p, x=x, y=y, blank='y', desc=f"Find y given x={x}, p={p}")
        problems.append(prob)
    return problems
