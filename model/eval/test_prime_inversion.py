import unittest
from random import randint

from sympy import randprime

from model.eval.prime_inversion import solve_modular_inverse


class TestModularInverse(unittest.TestCase):
    def test_modular_inverse(self, num_tests=20, num_range=(100, 10000)):
        passed = 0
        failed = 0
        for i in range(num_tests):
            # Select a random prime p
            p = randprime(num_range[0], num_range[1])
            # Pick a random x in 1..p-1
            x = randint(1, p-1)
            # Compute y such that x*y == 1 mod p
            y = solve_modular_inverse(p, x=x)
            # Verify
            assert (x * y) % p == 1, f"Test failed for p={p}, x={x}, y={y} (xy mod p = {x*y%p})"

            # Now, test recovery from (p,y) -> x
            rec_x = solve_modular_inverse(p, y=y)
            if rec_x == x % p:  # modular inverse sometimes wraps
                passed += 1
            else:
                print(f"FAIL (recovery x): p={p}, y={y}, real x={x}, rec_x={rec_x}")
                failed += 1
                continue

            # Similarly, test recovery from (p,x) -> y
            rec_y = solve_modular_inverse(p, x=x)
            if rec_y == y % p:
                passed += 1
            else:
                print(f"FAIL (recovery y): p={p}, x={x}, real y={y}, rec_y={rec_y}")
                failed += 1

        print(f"PASSED: {passed} tests, FAILED: {failed} tests out of {num_tests*2} cases.")


if __name__ == "__main__":
    unittest.main() 
