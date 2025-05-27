import pytest
from custom_types import Problem
from model.eval.prompts import create_prompt


class TestCreatePrompt:

    def test_create_prompt_with_blank_x(self):
        problem = Problem(prime=7, x=None, y=3, blank='x')
        result = create_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_prompt_with_blank_y(self):
        problem = Problem(prime=11, x=5, y=None, blank='y')
        result = create_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_prompt_with_blank_p(self):
        problem = Problem(prime=None, x=2, y=4, blank='p')
        result = create_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_prompt_invalid_blank_raises_error(self):
        problem = Problem(prime=13, x=2, y=8, blank='z')
        with pytest.raises(ValueError, match="invalid blank value z"):
            create_prompt(problem)

    @pytest.mark.parametrize(
        "prime,x,y,blank",
        [
            (17, None, 6, 'x'),
            (19, 4, None, 'y'),
            (None, 3, 9, 'p'),
        ]
    )
    def test_create_prompt_parametrized(self, prime, x, y, blank):
        problem = Problem(prime=prime, x=x, y=y, blank=blank)
        result = create_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_prompt_contains_problem_values_for_x_blank(self):
        problem = Problem(prime=23, x=None, y=7, blank='x')
        result = create_prompt(problem)
        assert '7' in result
        assert '23' in result

    def test_create_prompt_contains_problem_values_for_y_blank(self):
        problem = Problem(prime=29, x=8, y=None, blank='y')
        result = create_prompt(problem)
        assert '8' in result
        assert '29' in result

    def test_create_prompt_contains_problem_values_for_p_blank(self):
        problem = Problem(prime=None, x=5, y=12, blank='p')
        result = create_prompt(problem)
        assert '5' in result
        assert '12' in result
