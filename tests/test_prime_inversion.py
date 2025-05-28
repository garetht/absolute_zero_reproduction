import pytest
from model.eval.prime_inversion import solve_modular_inverse, generate_problems, PRIMES, is_prime


@pytest.mark.parametrize(
    "prob",
    generate_problems(100, PRIMES),
    ids=lambda p: (
            f"solve_x_p{p.prime}_y{p.y}" if p.blank == 'x' else f"solve_y_p{p.prime}_x{p.x}"
    )
)
def test_modular_inverse_problem(prob):
    if prob.blank == 'x':
        computed_x = solve_modular_inverse(prob.prime, x=None, y=prob.y)
        assert computed_x == prob.x, (
            f"Wanted x={prob.x}, got {computed_x} (p={prob.prime}, y={prob.y})"
        )
    else:
        computed_y = solve_modular_inverse(prob.prime, x=prob.x, y=None)
        assert computed_y == prob.y, (
            f"Wanted y={prob.y}, got {computed_y} (p={prob.prime}, x={prob.x})"
        )


class TestIsPrime:
    @pytest.mark.parametrize(
        "number,expected",
        [
            (num, num in PRIMES)
            for num in range(2, PRIMES[-1] + 1)
        ]
    )
    def test_is_prime_parametrized(self, number, expected):
        assert is_prime(number) is expected
