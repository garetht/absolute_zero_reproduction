import pytest
from custom_types import Problem, TaskType
from utils.string_formatting import create_solver_prompt


class TestCreatePrompt:

    def test_create_prompt_with_blank_x(self):
        problem = Problem(prime=7, x_list=[99], y_list=[3], task_type=TaskType.ABDUCTION)
        result = create_solver_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0
        assert '99' not in result

    def test_create_prompt_with_blank_y(self):
        problem = Problem(prime=11, x_list=[5], y_list=[99], task_type=TaskType.DEDUCTION)
        result = create_solver_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0
        assert '99' not in result

    def test_create_prompt_with_blank_p(self):
        problem = Problem(prime=99, x_list=[2], y_list=[4], task_type=TaskType.INDUCTION)
        result = create_solver_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0
        assert '99' not in result

    @pytest.mark.parametrize(
        "prime,x,y,task_type",
        [
            (17, 99, 6, TaskType.ABDUCTION),
            (19, 4, 99, TaskType.DEDUCTION),
            (99, 3, 9, TaskType.INDUCTION),
        ]
    )
    def test_create_prompt_parametrized(self, prime, x, y, task_type):
        problem = Problem(prime=prime, x_list=[x], y_list=[y], task_type=task_type)
        result = create_solver_prompt(problem)
        assert isinstance(result, str)
        assert len(result) > 0
        assert '99' not in result

    def test_create_prompt_contains_problem_values_for_x_blank(self):
        problem = Problem(prime=23, x_list=[99], y_list=[7], task_type=TaskType.ABDUCTION)
        result = create_solver_prompt(problem)
        assert '7' in result
        assert '23' in result
        assert '99' not in result

    def test_create_prompt_contains_problem_values_for_y_blank(self):
        problem = Problem(prime=29, x_list=[8], y_list=[99], task_type=TaskType.DEDUCTION)
        result = create_solver_prompt(problem)
        assert '8' in result
        assert '29' in result
        assert '99' not in result

    def test_create_prompt_contains_problem_values_for_p_blank(self):
        problem = Problem(prime=99, x_list=[5], y_list=[12], task_type=TaskType.INDUCTION)
        result = create_solver_prompt(problem)
        assert '5' in result
        assert '12' in result
        assert '99' not in result
