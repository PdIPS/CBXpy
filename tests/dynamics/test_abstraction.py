import cbx
import pytest
from abc import ABC, abstractmethod
import numpy as np

class test_abstract_dynamic(ABC):
    
    @pytest.fixture
    def dynamic(self):
        raise NotImplementedError()
    
    @pytest.fixture
    def f(self):
        return cbx.objectives.Quadratic()
    
    def test_torch_handeling(self, f, dynamic):
        '''Test if torch is correctly handled'''
        import torch
        x = torch.zeros((6,5,7))
        dyn = dynamic(f, x=x, max_it=2, array_mode='torch')
        dyn.optimize()
        assert dyn.x.shape == (6,5,7)