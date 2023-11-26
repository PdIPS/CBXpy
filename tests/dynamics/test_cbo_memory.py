from cbx.dynamics import CBOMemory
import pytest
from test_abstraction import test_abstract_dynamic
import numpy as np

class Test_cbo_memory(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return CBOMemory
    
    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5, term_args={'max_it':1})
        dyn.step()
        assert dyn.it == 1
        
    def test_update_best_cur_particle(self, f, dynamic):
        x = np.zeros((5,3,2))
        x[0, :,:] = np.array([[0.,0.], [2.,1.], [4.,5.]])
        x[1, :,:] = np.array([[8.,7.], [0.,1.], [2.,1.]])
        x[2, :,:] = np.array([[2.,5.], [0.,0.5], [2.,1.]])
        x[3, :,:] = np.array([[5.3,0.], [2.,1.], [0.,0.3]])
        x[4, :,:] = np.array([[0.,3.], [2.,1.], [0.,1.]])
        dyn = dynamic(f, x=x)
        dyn.update_best_cur_particle()
        best_cur_particle = np.array([[0.,0.], [0.,1.], [0.,0.5], [0.,0.3], [0.,1.]])
        
        assert np.allclose(dyn.best_cur_particle, best_cur_particle)