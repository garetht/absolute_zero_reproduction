def modinv(a, m):
    # Extended Euclidean Algorithm to find modular inverse
    # Returns inverse of a mod m, or raises ValueError if none exists
    a = a % m
    if a == 0:
        raise ValueError("No inverse exists for zero modulo m")
    t, newt = 0, 1
    r, newr = m, a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        raise ValueError(f"{a} has no inverse modulo {m}")
    if t < 0:
        t = t + m
    return t


def solve_modular_inverse(x=None, y=None, p=None):
    # Accepts any two of x, y, q (exactly one must be None)
    args = [x, y, p]
    if args.count(None) != 1:
        raise ValueError("Exactly one of x, y, q must be None")
    # All must be integers or None
    if not all(val is None or isinstance(val, int) for val in args):
        raise TypeError("Arguments must be integers or None")
    # Solving:
    ## 1. x missing
    if x is None:
        if y is None or q is None:
            raise ValueError("Two values must be provided")
        # Find x such that x*y ≡ 1 mod q ==> x ≡ y⁻¹ mod q
        inv = modinv(y, p)
        x = inv
        return x
    ## 2. y missing
    elif y is None:
        if x is None or p is None:
            raise ValueError("Two values must be provided")
        # x*y ≡ 1 mod q ==> y ≡ x⁻¹ mod q
        inv = modinv(x, p)
        y = inv
        return y
    ## 3. q missing
    elif p is None:
        if x is None or y is None:
            raise ValueError("Two values must be provided")
        # Find the minimal q > 1 such that x*y ≡ 1 mod q
        # That is, find q such that (x*y - 1) % q == 0  and q > 1
        diff = x * y - 1
        if diff == 0:
            return float('inf')  # Infinite possible q (all q>1)
        factors = [d for d in range(2, abs(diff) + 1) if diff % d == 0]
        if not factors:
            return None  # No modulus q > 1 makes x*y ≡ 1 mod q true (unless diff==0)
        return factors[0]  # Return the smallest possible modulus
    else:
        raise ValueError("Exactly one argument must be None")
