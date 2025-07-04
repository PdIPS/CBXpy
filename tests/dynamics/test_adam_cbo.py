from cbx.dynamics.adamcbo import AdamCBO
from test_cbo import Test_cbo
import pytest


class Test_adamcbo(Test_cbo):
    """
    Test class for AdamCBO dynamic.
    Inherits from Test_cbo to reuse tests for CBO dynamics.
    """

    @pytest.fixture
    def dynamic(self):
        return AdamCBO
    
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