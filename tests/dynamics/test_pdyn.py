import cbx
from cbx.dynamic.pdyn import ParticleDynamic
import pytest
import numpy as np
from test_abstraction import test_abstract_dynamic

class Test_pdyn(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return ParticleDynamic
    
    def test_term_crit_maxit(self, dynamic, f):
        '''Test termination criterion on max iteration'''
        dyn = dynamic(f, d=5, max_it=7)
        dyn.optimize()
        assert dyn.it == 7

    def test_term_crit_maxtime(self, dynamic, f):
        '''Test termination criterion on max time'''
        dyn = dynamic(f, d=5, max_time=0.1, dt=0.02)
        dyn.optimize()
        assert dyn.t == 0.1

    def test_no_given_x(self, dynamic, f):
        '''Test if x is correctly initialized'''
        dyn = dynamic(f, d=5, M=4, N=3)
        assert dyn.x.shape == (4,3,5)
        assert dyn.M == 4
        assert dyn.N == 3

    def test_given_x_1D(self, dynamic, f):
        '''Test if given x (1D) is correctly reshaped'''
        dyn = dynamic(f, x=np.zeros((7)), max_it=1)
        assert dyn.x.shape == (1,1,7)
        assert dyn.M == 1
        assert dyn.N == 1

    def test_given_x_2D(self, dynamic, f):
        '''Test if given x (2D) is correctly reshaped'''
        dyn = dynamic(f, x=np.zeros((5,7)), max_it=1)
        assert dyn.x.shape == (1,5,7)
        assert dyn.M == 1
        assert dyn.N == 5

    def test_opt_hist_and_output(self, dynamic, f):
        '''Test if optimization history is correctly saved and output is correct'''
        dyn = dynamic(f, x = np.zeros((6,5,7)), max_it=10)
        x, x_hist = dyn.optimize(save_particles=True, print_int=3)
        x_hist = np.array(x_hist)
        assert x_hist.shape == (4,6,5,7)
        assert x.shape == (6,7)
        assert dyn.x.shape == (6,5,7)


    

