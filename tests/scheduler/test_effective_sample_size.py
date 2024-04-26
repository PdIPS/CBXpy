import numpy as np
from cbx.scheduler import eff_sample_size_gap, bisection_solve


def test_alpha_to_zero():
    '''Test if effective sample size scheduler updates params correctly'''
    energy = np.random.normal((6,5,7))
    gap = eff_sample_size_gap(energy, 1.)
    alpha = bisection_solve(gap, 0*np.ones(6,), 100*np.ones(6,), max_it = 100, thresh = 1e-6)
    assert np.max(alpha) < 1e-1