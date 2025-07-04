from cbx.dynamics.regcombinationcbo import RegCombinationCBO
from test_cbo import Test_cbo
import pytest


class Test_regcombination_cbo(Test_cbo):
    """
    Test class for RegCombination dynamic.
    Inherits from Test_cbo to reuse tests for CBO dynamics.
    """

    @pytest.fixture
    def dynamic(self):
        return RegCombinationCBO
    
    def test_step(self, dynamic, f):
        # ToDo: Implement a test for the step function
        pass

    def test_step_batched(self, dynamic, f):
        # ToDo: Implement a test for the step function with batching
        pass

    def test_step_batched_partial(self, dynamic, f):
        # ToDo: Implement a test for the step function with batching
        pass

    def test_torch_handling(self, f, dynamic):
        # ToDo: Implement a test for handling torch tensors
        pass

    def test_optimization_performance(self, f, dynamic, opt_kwargs):
        # ToDo: Implement a test for optimization performance
        pass