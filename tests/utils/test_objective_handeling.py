import pytest
import numpy as np
from cbx.utils.objective_handling import _promote_objective, batched_objective_from_1D, batched_objective_from_2D

def test_f_dim_1D_handeling():
    '''Test if f_dim is correctly handeled for 1D'''
    f = lambda x: np.sum(x**2)
    f_dim = '1D'
    f_promote = _promote_objective(f, f_dim)
    x = np.random.uniform(-1,1,(6,5,7))
    res = np.array([f(x[i,j,:]) for i in range(6) for j in range(5)]).reshape(6,5)

    assert np.all(f_promote(x) == res)

def test_f_dim_2D_handeling():
    '''Test if f_dim is correctly handeled for 2D'''
    f = lambda x: np.sum(x**2, axis=-1)
    f_dim = '2D'
    f_promote = _promote_objective(f, f_dim)
    x = np.random.uniform(-1,1,(6,5,7))
    res = np.array([f(x[i,j,:]) for i in range(6) for j in range(5)]).reshape(6,5)

    assert np.all(f_promote(x) == res)

def test_f_dim_unknown():
    '''Test if f_dim raises error for unknown f_dim'''
    f = lambda x: np.sum(x**2)
    f_dim = '4D'

    with pytest.raises(ValueError):
        f_promote = _promote_objective(f, f_dim)
