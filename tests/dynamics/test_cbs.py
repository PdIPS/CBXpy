from cbx.dynamics import CBS
import pytest
from test_abstraction import test_abstract_dynamic

class Test_pso(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return CBS
    
    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5, max_it=1)
        dyn.step()
        assert dyn.it == 1