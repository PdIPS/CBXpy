from cbx.dynamics.mirrorcbo import (
    MirrorCBO, ProjectionBall, ProjectionHyperplane,
    mirror_dict
)
from test_cbo import Test_cbo
import numpy as np
import pytest

class Test_mirrorcbo(Test_cbo):
    """
    Test class for MirrorCBO dynamics.
    Inherits from Test_cbo to reuse tests for CBO dynamics.
    """

    @pytest.fixture
    def dynamic(self):
        return MirrorCBO
    
    def test_if_all_mirror_maps_are_loadable(self, dynamic, f):
        """
        Test if all mirror maps in mirror_dict are loadable.
        """
        for mm in mirror_dict.keys():
            dyn = dynamic(f, d=5, M=4, N=3, mirrormap=mm)
            dyn.step()

    def test_ProjectionBall(self, dynamic, f):
        """
        Test the ProjectionBall mirror map.
        """
        dyn = dynamic(f, d=5, M=4, N=3, mirrormap=ProjectionBall(radius=1.0))
        dyn.step()
        assert dyn.it > 0
        assert np.all(np.linalg.norm(dyn.x - dyn.mirrormap.center, axis=-1) <= 1 + 1e-5)

    def test_ProjectionHyperplane(self, dynamic, f):
        """
        Test the ProjectionHyperplane mirror map.
        """
        dyn = dynamic(f, d=5, M=4, N=3, mirrormap=ProjectionHyperplane(a=np.ones((5,))))
        dyn.step()
        assert dyn.it > 0
        assert np.all(np.abs(dyn.x @ np.ones((5,))) <= 1e-5)