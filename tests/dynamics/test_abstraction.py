import cbx
import pytest
#from abc import ABC

class test_abstract_dynamic():
    
    @pytest.fixture
    def dynamic(self):
        raise NotImplementedError()
    
    @pytest.fixture
    def f(self):
        return cbx.objectives.Quadratic()
    
    def test_torch_handling(self, f, dynamic):
        '''Test if torch is correctly handled'''
        import torch
        x = torch.zeros((6,5,7))
        dyn = dynamic(f, x=x, max_it=2, array_mode='torch')
        dyn.optimize()
        assert dyn.x.shape == (6,5,7)

    
    def test_eval_counting(self, f, dynamic):
        '''Test if evaluation counting is correct'''
        dyn = dynamic(f, d=5, M=7, N = 5, max_it=2)
        dyn.optimize()
        assert dyn.num_f_eval.shape == (7,)
        assert dyn.num_f_eval.sum() == dyn.f.eval_count