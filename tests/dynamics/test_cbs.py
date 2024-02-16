from cbx.dynamics import CBS
import pytest
from test_abstraction import test_abstract_dynamic
import numpy as np

class Test_CBS(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return CBS
    
    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5, max_it=1)
        dyn.step()
        assert dyn.it == 1

    def test_run(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5, max_it=3)
        dyn.run()
        assert dyn.it == 3

    def test_run_optimization(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5, max_it=2, mode='optimization')
        dyn.run()
        assert dyn.it == 2
        
    def test_multi_dim_domain(self, f, dynamic):
        x = np.ones((5,7,2,3,1))
        def g(x):
            return np.sum(x, axis=(2,3,4))**2
        
        with pytest.raises(Exception):
            dyn = dynamic(g, x=x, f_dim ='3D')