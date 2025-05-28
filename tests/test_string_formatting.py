import pytest
from custom_types import TaskType, Answer
from utils.string_formatting import extract_modular_equation, ModularEquation, validate_proposer_formatting_and_correctness_bulk


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
        assert result.x == 5
        assert result.y == 3
        assert result.p == 7

    def test_extract_modular_equation_mixed_variables_and_numbers(self):
        text = "a * 4 ≡ 1 (mod 17)"
        result = extract_modular_equation(text)
        assert result is not None
        assert result.x == "a"
        assert result.y == 4
        assert result.p == 17

    def test_extract_modular_equation_invalid_format(self):
        text = "x + y = 1 mod p"
        result = extract_modular_equation(text)
        assert result == ModularEquation(
            x='',
            y='',
            p=''
        )

    def test_extract_modular_equation_empty_string(self):
        text = ""
        result = extract_modular_equation(text)
        assert result == ModularEquation(
            x='',
            y='',
            p=''
        )

    def test_extract_modular_equation_no_match(self):
        text = "This is just some random text"
        result = extract_modular_equation(text)
        assert result == ModularEquation(
            x='',
            y='',
            p=''
        )

    def test_extract_modular_equation_partial_match(self):
        text = "x * y ≡ 2 (mod p)"
        result = extract_modular_equation(text)
        assert result == ModularEquation(
            x='',
            y='',
            p=''
        )

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
            ("12 * 5 ≡ 1 (mod 13)", 12, 5, 13),
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


class TestValidateFormattingAndCorrectnessBulk:
    def test_validate_formatting_and_correctness_bulk_abduction_valid(self):
        responses = ["5 * 3 ≡ 1 (mod 7)"]
        result = validate_proposer_formatting_and_correctness_bulk(responses, TaskType.ABDUCTION)
        assert len(result) == 1
        assert isinstance(result[0], Answer)

    def test_validate_formatting_and_correctness_bulk_deduction_valid(self):
        responses = ["3 * 5 ≡ 1 (mod 7)"]
        result = validate_proposer_formatting_and_correctness_bulk(responses, TaskType.DEDUCTION)
        assert len(result) == 1
        assert isinstance(result[0], Answer)

    def test_validate_formatting_and_correctness_bulk_induction_valid(self):
        responses = ["3 * 5 ≡ 1 (mod 7)"]
        result = validate_proposer_formatting_and_correctness_bulk(responses, TaskType.INDUCTION)
        assert len(result) == 1
        assert isinstance(result[0], Answer)

    def test_validate_formatting_and_correctness_bulk_empty_responses(self):
        responses = []
        result = validate_proposer_formatting_and_correctness_bulk(responses, TaskType.ABDUCTION)
        assert len(result) == 0

    def test_validate_formatting_and_correctness_bulk_invalid_formatting(self):
        responses = ["invalid equation format"]
        result = validate_proposer_formatting_and_correctness_bulk(responses, TaskType.ABDUCTION)
        assert len(result) == 1

    def test_validate_formatting_and_correctness_bulk_multiple_responses(self):
        responses = ["3 * 5 ≡ 1 (mod 7)", "2 * 4 ≡ 1 (mod 7)"]
        result = validate_proposer_formatting_and_correctness_bulk(responses, TaskType.ABDUCTION)
        assert len(result) == 2
