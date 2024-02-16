import cbx
from cbx.dynamics.pdyn import ParticleDynamic, CBXDynamic
import pytest
import numpy as np
from test_abstraction import test_abstract_dynamic
from cbx.utils.objective_handling import cbx_objective_fh
import scipy as scp

class Test_pdyn(test_abstract_dynamic):
    
    @pytest.fixture
    def dynamic(self):
        return ParticleDynamic
    
    def test_term_crit_maxit(self, dynamic, f):
        '''Test termination criterion on max iteration'''
        dyn = dynamic(f, d=5, max_it=3)
        dyn.optimize()
        assert dyn.it == 3

    def test_no_given_x(self, dynamic, f):
        '''Test if x is correctly initialized'''
        dyn = dynamic(f, d=5, M=4, N=3)
        assert dyn.x.shape == (4,3,5)
        assert dyn.M == 4
        assert dyn.N == 3

    def test_given_x_1D(self, dynamic, f):
        '''Test if given x (1D) is correctly reshaped'''
        dyn = dynamic(f, x=np.zeros((7)), max_it=1)
        assert dyn.x.shape == (1,1,7)
        assert dyn.M == 1
        assert dyn.N == 1

    def test_given_x_2D(self, dynamic, f):
        '''Test if given x (2D) is correctly reshaped'''
        dyn = dynamic(f, x=np.zeros((5,7)), max_it=1)
        assert dyn.x.shape == (1,5,7)
        assert dyn.M == 1
        assert dyn.N == 5

    def test_opt_and_output(self, dynamic, f):
        '''Test if optimization history is correctly saved and output is correct'''
        dyn = dynamic(f, x = np.zeros((6,5,7)), 
                      max_it=10)
        x = dyn.optimize()
        assert x.shape == (6,7)
        assert dyn.x.shape == (6,5,7)

    def test_f_wrong_dims(self, dynamic):
        '''Test if f_dim raises error for wrong dimensions'''
        def f(x): return x
        f_dim = '1D'
        x = np.random.uniform(-1,1,(6,5,7))

        with pytest.raises(ValueError):
            dynamic(f, x=x, f_dim=f_dim)
            
    def test_torch_handling(self, f, dynamic):
        '''Test if torch is correctly handled'''
        import torch
        x = torch.zeros((6,5,7))
        
        @cbx_objective_fh
        def g(x):
            return torch.sum(x, dim=-1)
        
        def norm_torch(x, axis, **kwargs):
            return torch.linalg.norm(x, dim=axis, **kwargs)
        
        dyn = dynamic(g, 
                      f_dim = '3D',
                      x=x, 
                      max_it=2,
                      norm=norm_torch,
                      copy=torch.clone,
                      normal=torch.normal
                      )
        dyn.optimize()
        assert dyn.x.shape == (6,5,7)
  
        
def test_mat_sqrt():
    M=6
    A = np.random.uniform(size=(M,7,7))
    A = (A@A.transpose(0,2,1))
    C = cbx.dynamics.pdyn.compute_mat_sqrt(A) 
    CC = np.zeros_like(C)
    for m in range(M):
        CC[m,...] = scp.linalg.sqrtm(A[m,...])
    assert np.allclose(C, CC)
    

class Test_cbx(test_abstract_dynamic):
    @pytest.fixture
    def dynamic(self):
        return CBXDynamic
    
    def test_apply_cov_mat(self, f, dynamic):
        M=7
        d=4
        N=12
        dyn = dynamic(f, M=M, d=d, N=N, noise='covariance')
        A = np.random.uniform(size=(M,d,d))
        dyn.Cov_sqrt = A.copy()
        
        z = np.random.uniform(low=-1.,high=1., size=(M,N,d))
        
        g = np.zeros_like(z)
        for m in range(M):
            for n in range(N):
                g[m,n,:] = A[m,...]@z[m,n,:]
                
        gg = dyn.noise_callable.apply_cov_sqrt(dyn.Cov_sqrt,z)
        assert np.allclose(g, gg)
        
        
    def test_update_cov(self, f, dynamic):
        M=7
        d=4
        N=12
        dyn = dynamic(f, M=M, d=d, N=N)
        dyn.consensus, energy = dyn.compute_consensus()
        dyn.energy = energy
        dyn.drift = dyn.x - dyn.consensus
        dyn.update_covariance()
        
        Cov = np.zeros((M,d,d))
        for m in range(M):
            C = np.zeros((N, d,d))
            denom = 0.
            for n in range(N):
                drift = dyn.x[m,n,:] - dyn.consensus[m, 0, :]
                factor = np.exp(-dyn.alpha[m] * f(dyn.x[m,n,:]))
                denom += factor
                C[n,...] = np.outer(drift,drift) * factor
        
            Cov[m,...] = np.sum(C,axis=0)/denom
            
        dyn_Cov = dyn.Cov_sqrt@dyn.Cov_sqrt
        assert np.allclose(dyn_Cov, Cov)
            
            
        
        
        
        

    

