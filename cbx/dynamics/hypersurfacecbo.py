from cbx.dynamics import CBO
import numpy as np
from scipy.special import logsumexp as logsumexp_scp

#%%
class signedDistance:
    def get_proj_matrix(self, v):
        grad = self.grad(v)
        #outer = np.einsum('...i,...j->...ij',grad,grad)
        outer = grad[..., None,:]*grad[..., None]
        return np.eye(v.shape[-1]) - outer
    
    def proj_tangent_space(self, x, z):
        g = self.grad(x)
        return z - g * (g*z).sum(axis=-1, keepdims=True)
    
    
class sphereDistance(signedDistance):
    def grad(self, v):
        return v/np.linalg.norm(v,axis=-1, keepdims=True)
    
    def Laplacian(self, v):
        d = v.shape[-1]
        return (d-1)/np.linalg.norm(v, axis=-1, keepdims=True)
    
    def proj(self, x):
        return x/np.linalg.norm(x, axis=-1, keepdims=True)
    
class StiefelDistance(signedDistance):
    def __init__(self, nk= None):
        self.nk = nk
        
    def grad(self, v):
        return v
    
    def Laplacian(self, v):
        return (2*self.nk[0] - self.nk[1] - 1)/2
    
    def proj(self, x):
        U, _, Vh = np.linalg.svd(x.reshape(-1, self.nk[0], self.nk[1]), full_matrices=False)
        return (U@Vh).reshape(x.shape)
    
    def proj_tangent_space(self, x, z):
        X, Z = (y.reshape(-1, self.nk[0], self.nk[1]) for y in (x,z))
        X = x.reshape(-1, self.nk[0], self.nk[1])
        
        return (Z - 0.5 * X@(np.moveaxis(Z, -1,-2)@X + np.moveaxis(X, -1,-2)@Z)).reshape(x.shape)
    
class planeDistance(signedDistance):
    def __init__(self, a=0, b=1.):
        self.a = a
        self.norm_a = np.linalg.norm(a, axis=-1)
        self.b = b
        
    def grad(self, v):
        return self.a/self.norm_a * np.ones_like(v)
    
    def Laplacian(self, v):
        return 0
    
    def proj(self, y):
        return y - ((self.a * y).sum(axis=-1, keepdims=True) - self.b)/(self.norm_a**2) * self.a

        
sdf_dict = {'sphere': sphereDistance, 
            'plane': planeDistance,
            'Stiefel': StiefelDistance
            }    

def get_sdf(sdf):
    if sdf is None:
        return sphereDistance()
    else:
        return sdf_dict[sdf['name']](**{k:v for k,v in sdf.items() if k not in ['name']})


def compute_consensus_rescale(energy, x, alpha):
    emin = np.min(energy, axis=-1, keepdims=True)
    weights = - alpha * (energy - emin)
    coeff_expan = tuple([Ellipsis] + [None for i in range(x.ndim-2)])
    coeffs = np.exp(weights - logsumexp_scp(weights, axis=-1, keepdims=True))[coeff_expan]

    return (x * coeffs).sum(axis=1, keepdims=True), energy


    
    
class HyperSurfaceCBO(CBO):
    '''
    Implements Hypersurface CBO described in the following papers:
    [1] Massimo Fornasier, Hui Huang, Lorenzo Pareschi, Philippe Sünnen
    "Consensus-Based Optimization on Hypersurfaces: Well-Posedness and Mean-Field Limit"
    [2] Massimo Fornasier, Hui Huang, Lorenzo Pareschi, Philippe Sünnen
    "Consensus-Based Optimization on the Sphere: Convergence to Global Minimizers and Machine Learning"

    Parameters
    ----------
    f : callable
        The objective function to be minimized.
    noise : str, optional
        Type of noise to be used, either 'isotropic' or 'anisotropic'. Default is 'isotropic'.
    sdf : dict or None, optional
        Signed distance function to be used for the hypersurface. If None, a sphere distance is used.
        The dictionary should contain the name of the distance function and any additional parameters.
    **kwargs : additional keyword arguments
        Additional parameters for the CBO algorithm, such as `lamda`, `sigma`, etc.
    '''

    def __init__(self, f, noise = 'isotropic', sdf = None, **kwargs):
        super().__init__(
            f, 
            compute_consensus = compute_consensus_rescale,
            noise = noise,
            **kwargs
        )
        self.sdf = get_sdf(sdf)
        if noise == 'anisotropic':
            self.noise_constr = self.noise_constr_aniso
        else:
            self.noise_constr = self.noise_constr_iso
        
    def inner_step(self,):
        self.compute_consensus() # compute consensus, sets self.energy and self.consensus
        self.drift = self.x - self.consensus
        
        # compute relevant terms for update
        # dir_drift = self.x + self.dt * apply_proj(Px, self.consensus)
        # noise     = self.sigma * apply_proj(Px, self.noise())
        
        # perform addition before matrix multiplaction to save time
        drift_and_noise = (
            self.x + 
            self.sdf.proj_tangent_space(
                self.x, 
                self.dt * self.consensus + 
                self.sigma * self.noise())
        )
        
        x_tilde = drift_and_noise - self.noise_constr()
        self.x = self.sdf.proj(x_tilde)
        
    def noise_constr_iso(self,):
        return (
            self.dt * self.sigma**2/2 * self.sdf.Laplacian(self.x) *
            np.linalg.norm(self.drift, axis=-1, keepdims=True)**2 * 
            # note that in the implementation here:
            # https://github.com/PhilippeSu/KV-CBO/blob/master/v2/kvcbo/KuramotoIteration.m
            # norm(drift)**2 is used instead of drift**2
            #self.drift**2 *
            self.sdf.grad(self.x)
        )
    
    def noise_constr_aniso(self,):
        return (self.dt * self.sigma**2)/2 * (
            np.linalg.norm(self.drift, axis=-1, keepdims=True) ** 2 +
            self.drift ** 2 -
            2 * np.linalg.norm(self.drift * self.x, axis=-1, keepdims=True)**2
            ) * self.x