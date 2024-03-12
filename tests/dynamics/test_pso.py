from cbx.dynamics import PSO
import pytest
from test_abstraction import test_abstract_dynamic

class Test_pso(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return PSO
    
    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5)
        dyn.step()
        assert dyn.it == 1