import pytest
from model.eval.prime_inversion import solve_modular_inverse, generate_problems

PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
    113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
    241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
    383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521,
    523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617
]


@pytest.mark.parametrize("n", [100])
def test_modular_inverse_with_generated_problems(n):
    problems = generate_problems(n, PRIMES)
    for idx, prob in enumerate(problems):
        if prob.blank == 'x':
            computed_x = solve_modular_inverse(prob.prime, x=None, y=prob.y)
            assert computed_x == prob.x, (
                f"Problem {idx}: Wanted x={prob.x}, got {computed_x} (p={prob.prime}, y={prob.y})"
            )
        else:
            computed_y = solve_modular_inverse(prob.prime, x=prob.x, y=None)
            assert computed_y == prob.y, (
                f"Problem {idx}: Wanted y={prob.y}, got {computed_y} (p={prob.prime}, x={prob.x})"
            )
