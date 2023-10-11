from cbx.dynamics.cbo import CBO
import pytest
import numpy as np
from test_abstraction import test_abstract_dynamic
from cbx.utils.objective_handling import cbx_objective_fh

class Test_cbo(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return CBO
    
    def test_term_crit_energy(self, dynamic, f):
        '''Test termination criterion on energy'''
        dyn = dynamic(f, x=np.zeros((3,5,7)), energy_tol=1e-6, max_it=10)
        dyn.optimize()
        assert dyn.it == 1

    def test_term_crit_maxeval(self, dynamic, f):
        '''Test termination criterion on max function evaluations'''
        dyn = dynamic(f, d=5, M=4, N=3, max_eval=6, max_it=10)
        dyn.optimize()
        assert np.all(dyn.num_f_eval == np.array([6,6,6,6]))

    def test_mean_compute(self, dynamic, f):
        '''Test if mean is correctly computed'''
        x = np.random.uniform(-1,1,(3,5,7))
        dyn = dynamic(f, x=x)
        dyn.step()
        mean = np.zeros((3,1,7))

        for j in range(x.shape[0]):
            loc_mean = 0
            loc_denom = 0
            for i in range(x.shape[1]):
                loc_mean += np.exp(-dyn.alpha * f(x[j,i,:])) * x[j,i,:]
                loc_denom += np.exp(-dyn.alpha * f(x[j,i,:]))
            mean[j,...] = loc_mean / loc_denom

        assert np.allclose(dyn.consensus, mean)
    
    def test_mean_compute_batched(self, dynamic, f):
        '''Test if batched mean is correctly computed'''
        x = np.random.uniform(-1,1,(3,5,7))
        dyn = dynamic(f, x=x, batch_size=2)
        dyn.step()
        mean = np.zeros((3,1,7))
        ind = dyn.consensus_idx[1]

        for j in range(x.shape[0]):
            loc_mean = 0
            loc_denom = 0
            for i in range(ind.shape[1]):
                loc_mean += np.exp(-dyn.alpha * f(x[j,ind[j, i],:])) * x[j,ind[j, i],:]
                loc_denom += np.exp(-dyn.alpha * f(x[j,ind[j, i],:]))
            mean[j,...] = loc_mean / loc_denom

        assert np.allclose(dyn.consensus, mean)

    def test_step(self, dynamic, f):
        '''Test if step is correctly performed'''
        x = np.random.uniform(-1,1,(3,5,7))
        delta = np.random.uniform(-1,1,(3,5,7))
        def noise():
            return delta
        with pytest.warns():
            dyn = dynamic(f, x=x, noise=noise)
        dyn.step()
        x_new = x - dyn.lamda * dyn.dt * (x - dyn.consensus) + dyn.sigma * delta
        assert np.allclose(dyn.x, x_new)

    def test_step_batched(self, dynamic, f):
        '''Test if batched step is correctly performed'''
        x = np.random.uniform(-1,1,(3,5,7))
        delta = np.random.uniform(-1,1,(3,5,7))
        def noise():
            return delta
        with pytest.warns():
            dyn = dynamic(f, x=x, noise=noise, batch_size=2)
        dyn.step()
        x_new = x - dyn.lamda * dyn.dt * (x - dyn.consensus) + dyn.sigma * delta
        assert np.allclose(dyn.x, x_new)

    def test_step_batched_partial(self, dynamic, f):
        '''Test if partial batched step is correctly performed'''
        x = np.random.uniform(-1,1,(3,5,7))

        dyn = dynamic(f, x=x, batch_size=2, batch_partial=True)
        dyn.step()
        ind = dyn.particle_idx[1]
        for j in range(x.shape[0]):
            for i in range(ind.shape[1]):
                x[j, ind[j,i], :] = x[j, ind[j,i], :] - dyn.lamda * dyn.dt * (x[j, ind[j,i], :] - dyn.consensus[j, 0, :]) + dyn.s[j,i,:]
            
        assert np.allclose(dyn.x, x)
        
    def test_torch_handling(self, f, dynamic):
        '''Test if torch is correctly handled'''
        import torch
        x = torch.zeros((6,5,7))
        
        @cbx_objective_fh
        def g(x):
            return torch.sum(x, dim=-1)
        
        dyn = dynamic(g, x=x, max_it=2, array_mode='torch')
        dyn.optimize()
        assert dyn.x.shape == (6,5,7)