import cbx
import pytest
import numpy as np

class test_abstract_dynamic():
    
    @pytest.fixture
    def dynamic(self):
        raise NotImplementedError()
    
    @pytest.fixture
    def f(self):
        return cbx.objectives.Quadratic()

    
    def test_eval_counting(self, f, dynamic):
        '''Test if evaluation counting is correct'''
        f.reset()
        dyn = dynamic(f, d=5, M=7, N=5, max_it=3, 
                      check_f_dims=False)
        dyn.optimize()
        
        assert dyn.num_f_eval.shape == (7,)
        assert dyn.num_f_eval.sum() == dyn.f.num_eval
        
            
    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5, max_it=1)
        dyn.step()
        assert dyn.it == 1
        
    def test_multi_dim_domain(self, f, dynamic):
        x = np.ones((5,7,2,3,1))
        def g(x):
            return np.sum(x, axis=(2,3,4))**2
        
        dyn = dynamic(g, x=x, f_dim ='3D')
        assert dyn.M == 5
        assert dyn.N == 7
        dyn.step()
        assert dyn.d == (2,3,1)