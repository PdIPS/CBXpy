import pytest
import numpy as np
from cbx.utils.particle_init import init_particles

def test_particle_init_uniform():
    '''Test if particles are correctly initialized'''
    shape = (2,3,5,7)
    x = init_particles(shape=shape, x_min=-1, x_max=1, method="uniform")
    assert x.shape == (2,3,5,7)
    assert np.all(x >= -1)
    assert np.all(x <= 1)

@pytest.mark.parametrize("in_shape, out_shape", [((3,5,7), (3,5,7)), ((5,7), (1,5,7))])
def test_particle_init_normal(in_shape, out_shape):
    '''Test if particles are correctly initialized'''
    x = init_particles(shape=in_shape, method="normal")
    assert x.shape == out_shape

def test_particle_init_normal_wrong_shape():
    '''Test if exception is raised when normal initialization is used for 2D particles'''
    with pytest.raises(RuntimeError):
        init_particles(shape=(3,), method="normal")

def test_particle_init_unknown_method():
    '''Test if exception is raised when unknown method is specified'''
    with pytest.raises(RuntimeError):
        init_particles(shape=(3,5,7), method="unknown")
