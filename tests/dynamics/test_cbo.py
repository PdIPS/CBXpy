from cbx.dynamics.cbo import CBO
import pytest
import numpy as np
from test_abstraction import test_abstract_dynamic
from cbx.utils.termination import max_it_term, energy_tol_term, max_eval_term, max_time_term

class Test_cbo(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return CBO
    
    def test_term_crit_energy(self, dynamic, f):
        '''Test termination criterion on energy'''
        dyn = dynamic(f, x=np.zeros((3,5,7)), term_criteria=[energy_tol_term(1e-6), max_it_term(10)])
        dyn.optimize()
        assert dyn.it == 1
        
    def test_term_crit_maxtime(self, dynamic, f):
        '''Test termination criterion on max time'''
        dyn = dynamic(f, d=5, term_criteria=[max_time_term(0.1)], dt=0.02)
        dyn.optimize()
        assert dyn.t == 0.1

    def test_term_crit_maxeval(self, dynamic, f):
        '''Test termination criterion on max function evaluations'''
        dyn = dynamic(f, d=5, M=4, N=3, term_criteria=[max_eval_term(6), max_it_term(10)])
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
                loc_mean += np.exp(-dyn.alpha[j] * f(x[j,i,:])) * x[j,i,:]
                loc_denom += np.exp(-dyn.alpha[j] * f(x[j,i,:]))
            mean[j,...] = loc_mean / loc_denom

        assert np.allclose(dyn.consensus, mean)
    
    def test_mean_compute_batched(self, dynamic, f):
        '''Test if batched mean is correctly computed'''
        x = np.random.uniform(-1,1,(3,5,7))
        dyn = dynamic(f, x=x, batch_args={'size':2})
        dyn.step()
        mean = np.zeros((3,1,7))
        ind = dyn.consensus_idx[1]

        for j in range(x.shape[0]):
            loc_mean = 0
            loc_denom = 0
            for i in range(ind.shape[1]):
                loc_mean += np.exp(-dyn.alpha[j] * f(x[j,ind[j, i],:])) * x[j,ind[j, i],:]
                loc_denom += np.exp(-dyn.alpha[j] * f(x[j,ind[j, i],:]))
            mean[j,...] = loc_mean / loc_denom

        assert np.allclose(dyn.consensus, mean)

    def test_step(self, dynamic, f):
        '''Test if step is correctly performed'''
        x = np.random.uniform(-1,1,(3,5,7))
        delta = np.random.uniform(-1,1,(3,5,7))
        def noise(dyn):
            return delta

        dyn = dynamic(f, x=x, noise=noise)
        dyn.step()
        x_new = x - dyn.lamda * dyn.dt * (x - dyn.consensus) + dyn.sigma * delta
        assert np.allclose(dyn.x, x_new)

    def test_step_batched(self, dynamic, f):
        '''Test if batched step is correctly performed'''
        x = np.random.uniform(-1,1,(3,5,7))
        delta = np.random.uniform(-1,1,(3,5,7))
        def noise(dyn):
            return delta

        dyn = dynamic(f, x=x, noise=noise, batch_args={'size':2, 'partial':False})
        dyn.step()
        x_new = x - dyn.lamda * dyn.dt * (x - dyn.consensus) + dyn.sigma * delta
        assert np.allclose(dyn.x, x_new)

    def test_step_batched_partial(self, dynamic, f):
        '''Test if partial batched step is correctly performed'''
        x = np.random.uniform(-1,1,(3,5,7))
        class noise:
            def __call__(self, dyn):
                self.s = dyn.sampler(size=dyn.drift.shape) * dyn.drift
                return self.s
        N = noise()

        dyn = dynamic(f, x=x, noise=N, batch_args={'size':2, 'partial':True})
        dyn.step()
        ind = dyn.particle_idx[1]
        for j in range(x.shape[0]):
            for i in range(ind.shape[1]):
                x[j, ind[j,i], :] = x[j, ind[j,i], :] -\
                    dyn.lamda * dyn.dt * (x[j, ind[j,i], :] - dyn.consensus[j, 0, :])\
                        + dyn.sigma * N.s[j,i,:]
            
        assert np.allclose(dyn.x, x)
        
    def test_torch_handling(self, f, dynamic):
        '''Test if torch is correctly handled'''
        import torch
        from cbx.utils.torch_utils import to_torch_dynamic
        x = torch.randn(size=(6,25,3))
        
        
        def g(x):
            return torch.sum(x, dim=-1)**2
        
        def norm_torch(x, axis, **kwargs):
            return torch.linalg.norm(x, dim=axis, **kwargs)

        dyn = to_torch_dynamic(dynamic)(g,
                      f_dim = '3D',
                      x=x,
                      max_it=15,)
        dyn.optimize()
        assert dyn.x.shape == x.shape and (dyn.x is not x)
        
    def test_update_best_cur_particle(self, f, dynamic):
        x = np.zeros((5,3,2))
        x[0, :,:] = np.array([[0.,0.], [2.,1.], [4.,5.]])
        x[1, :,:] = np.array([[8.,7.], [0.,1.], [2.,1.]])
        x[2, :,:] = np.array([[2.,5.], [0.,0.5], [2.,1.]])
        x[3, :,:] = np.array([[5.3,0.], [2.,1.], [0.,0.3]])
        x[4, :,:] = np.array([[0.,3.], [2.,1.], [0.,1.]])
        dyn = dynamic(f, x=x)
        dyn.step()
        best_cur_particle = np.array([[0.,0.], [0.,1.], [0.,0.5], [0.,0.3], [0.,1.]])
        
        assert np.allclose(dyn.best_cur_particle, best_cur_particle)

    def test_anisotropic_noise(self, f, dynamic):
        dyn = dynamic(f, d=3, noise='anisotropic')
        dyn.step()
        s = dyn.noise()
        assert s.shape == dyn.x.shape

    def test_heavi_side(self, f, dynamic):
        dyn = dynamic(f, d=3, N=4, M=2, correction='heavi_side')
        dyn.step()
        assert dyn.x.shape == (2, 4, 3)

    def test_heavi_side_reg(self, f, dynamic):
        dyn = dynamic(f, d=3, N=4, M=2, correction='heavi_side_reg')
        dyn.step()
        assert dyn.x.shape == (2, 4, 3)