from cbx.dynamics import PSO
import pytest
import numpy as np
from test_abstraction import test_abstract_dynamic

class Test_pso(test_abstract_dynamic):

    @pytest.fixture
    def dynamic(self):
        return PSO

    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5)
        dyn.step()
        assert dyn.it == 1

    def test_consensus_uses_historical_best(self, f, dynamic):
        """compute_consensus must be evaluated on y (historical best), not x."""
        np.random.seed(0)
        x = np.random.uniform(-1, 1, (1, 10, 3))
        dyn = dynamic(f, x=x)

        # Drive y far from x so the two inputs to compute_consensus are distinct
        dyn.y = dyn.y * 0 + 5.0

        mind = dyn.consensus_idx
        c_from_y = dyn.compute_consensus(dyn.y[mind], dyn.energy[mind])
        c_from_x = dyn.compute_consensus(dyn.x[mind], dyn.energy[mind])

        # Consensus from y (all 5s) should be close to 5; from x (near 0) should not
        assert not np.allclose(c_from_y, c_from_x), (
            "compute_consensus returned same result for y and x — "
            "x_batch argument is being ignored"
        )