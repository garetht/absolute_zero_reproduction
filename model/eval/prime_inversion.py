def solve_modular_inverse(p: int, x=None, y=None):
    """
    Solves for the unknown in xy ≡ 1 mod p, where p is prime, and one of x or y is given.

    Args:
        p (int): A prime modulus.
        x (int, optional): The value of x. If None, y must be given.
        y (int, optional): The value of y. If None, x must be given.

    Returns:
        int: The value of the unknown variable (mod p).
    """
    if (x is None and y is None) or (x is not None and y is not None):
        raise ValueError("Exactly one of x or y must be provided.")
    if x is not None:
        # Compute modular inverse of x mod p
        result = pow(x, -1, p)
        print(f"Given x = {x}, solving for y such that x*y ≡ 1 mod {p}: y = {result}")
        return result
    else:
        # Compute modular inverse of y mod p
        result = pow(y, -1, p)
        print(f"Given y = {y}, solving for x such that x*y ≡ 1 mod {p}: x = {result}")
        return result
