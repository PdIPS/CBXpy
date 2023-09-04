import pytest
import numpy as np
from cbx.utils.numpy_torch_comp import copy_particles
import torch

def test_torch_handeling():
    '''Test if torch is correctly handled'''
    x = np.zeros((6,5,7))
    x_copy = copy_particles(x, mode='numpy')
    x_torch = torch.zeros((6,5,7))
    x_torch_copy = copy_particles(x_torch, mode='torch')
    
    assert np.all(x == x_copy)
    assert torch.all(x_torch == x_torch_copy)
    assert not (x is x_copy)
    assert not (x_torch is x_torch_copy)
