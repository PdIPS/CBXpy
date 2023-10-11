from cbx.dynamics import CBOMemory
import pytest
from test_abstraction import test_abstract_dynamic

class Test_cbo_memory(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return CBOMemory
    
    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5, max_it=1)
        dyn.step()
        assert dyn.it == 1