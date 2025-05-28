import pytest
from utils.string_formatting import extract_modular_equation, ModularEquation


class TestExtractModularEquation:
    def test_extract_modular_equation_valid_input(self):
        text = "x * y ≡ 1 (mod p)"
        result = extract_modular_equation(text)
        assert result is not None
        assert result.x == "x"
        assert result.y == "y"
        assert result.p == "p"

    def test_extract_modular_equation_with_numbers(self):
        text = "5 * 3 ≡ 1 (mod 7)"
        result = extract_modular_equation(text)
        assert result is not None
        assert result.x == "5"
        assert result.y == "3"
        assert result.p == "7"

    def test_extract_modular_equation_mixed_variables_and_numbers(self):
        text = "a * 4 ≡ 1 (mod 17)"
        result = extract_modular_equation(text)
        assert result is not None
        assert result.x == "a"
        assert result.y == "4"
        assert result.p == "17"

    def test_extract_modular_equation_invalid_format(self):
        text = "x + y = 1 mod p"
        result = extract_modular_equation(text)
        assert result is None

    def test_extract_modular_equation_empty_string(self):
        text = ""
        result = extract_modular_equation(text)
        assert result is None

    def test_extract_modular_equation_no_match(self):
        text = "This is just some random text"
        result = extract_modular_equation(text)
        assert result is None

    def test_extract_modular_equation_partial_match(self):
        text = "x * y ≡ 2 (mod p)"
        result = extract_modular_equation(text)
        assert result is None

    def test_extract_modular_equation_with_surrounding_text(self):
        text = "The equation x * y ≡ 1 (mod p) is important."
        result = extract_modular_equation(text)
        assert result is not None
        assert result.x == "x"
        assert result.y == "y"
        assert result.p == "p"

    @pytest.mark.parametrize(
        "text,expected_x,expected_y,expected_p",
        [
            ("a * b ≡ 1 (mod c)", "a", "b", "c"),
            ("12 * 5 ≡ 1 (mod 13)", "12", "5", "13"),
            ("var1 * var2 ≡ 1 (mod prime)", "var1", "var2", "prime"),
        ]
    )
    def test_extract_modular_equation_parametrized(self, text, expected_x, expected_y, expected_p):
        result = extract_modular_equation(text)
        assert result is not None
        assert result.x == expected_x
        assert result.y == expected_y
        assert result.p == expected_p

    def test_extract_modular_equation_returns_modular_equation_type(self):
        text = "x * y ≡ 1 (mod p)"
        result = extract_modular_equation(text)
        assert isinstance(result, ModularEquation)
