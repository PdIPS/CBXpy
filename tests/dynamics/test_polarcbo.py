from cbx.dynamics import PolarCBO
import pytest
from test_abstraction import test_abstract_dynamic
import numpy as np

class Test_polarcbo(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return PolarCBO
    
    def test_step_eval(self, f, dynamic):
        dyn = dynamic(f, d=5, M=7, N=5)
        dyn.step()
        assert dyn.it == 1

    def test_Gaussian_kernel(self, f, dynamic):
        dyn = dynamic(f, d=3, M=7, N=5, kernel='Gaussian')
        dyn.step()
        eval = dyn.kernel(dyn.x[:,None,...], dyn.x[:,:,None,...])
        assert dyn.consensus.shape == dyn.x.shape
        assert eval.shape == (7,5,5)

    def test_Laplace_kernel(self, f, dynamic):
        dyn = dynamic(f, d=3, M=7, N=5, kernel='Laplace')
        dyn.step()
        eval = dyn.kernel(dyn.x[:,None,...], dyn.x[:,:,None,...])
        assert dyn.consensus.shape == dyn.x.shape
        assert eval.shape == (7,5,5)

    def test_Constant_kernel(self, f, dynamic):
        dyn = dynamic(f, d=3, M=7, N=5, kernel='Constant')
        eval = dyn.kernel(dyn.x[:,None,...], dyn.x[:,:,None,...])
        dyn.step()
        assert dyn.consensus.shape == dyn.x.shape
        assert eval.shape == (7,5,5)

    def test_IQ_kernel(self, f, dynamic):
        dyn = dynamic(f, d=3, M=7, N=5, kernel='InverseQuadratic')
        eval = dyn.kernel(dyn.x[:,None,...], dyn.x[:,:,None,...])
        dyn.step()
        assert dyn.consensus.shape == dyn.x.shape
        assert eval.shape == (7,5,5)

    def test_Taz_kernel(self, f, dynamic):
        dyn = dynamic(f, d=3, M=7, N=5, kernel='Taz')
        dyn.step()
        assert dyn.consensus.shape == dyn.x.shape

    def test_consensus_shape(self, f, dynamic):
        dyn = dynamic(f, d=3, M=7, N=5)
        dyn.step()
        assert dyn.consensus.shape == (7, 5, 3)

    def test_consensus_value(self, f, dynamic):
        dyn = dynamic(f, d=3, M=7, N=5, kernel_factor_mode='const')
        c = np.zeros(dyn.x.shape)
        dd = dyn.kernel(dyn.x[:,None,...], dyn.x[:,:,None,...])
        d = np.zeros((7,5,5))
        for m in range(dyn.M):
            for n in range(dyn.N):
                nom = np.zeros(dyn.d)
                denom = 0.
                for nn in range(dyn.N):
                    d[m, n, nn] = np.exp(-1/(2*dyn.kernel.kappa**2) *  ((dyn.x[m,n,...] - dyn.x[m,nn,...])**2).sum())
                    w = d[m, n, nn] * np.exp(-dyn.alpha[m] * dyn.f(dyn.x[m,nn,...]))[0,0]
                    nom += w * dyn.x[m,nn,:]
                    denom += w
                c[m,n,:] = nom / denom
        dyn.compute_consensus()
        assert np.allclose(dd, d)
        assert np.allclose(dyn.consensus, c)


    