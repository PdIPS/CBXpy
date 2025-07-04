from cbx.dynamics.cbo import CBO, cbo_update
from cbx.utils.history import track
import numpy as np
import warnings
#%%
def raise_NIE(cls_name, f_name):
    raise NotImplementedError(
        'The class ' + cls_name + 
        ' does not implement the function ' + f_name
        )
    
def Bregman_distance(F, p, q):
    return F(p) - F(q) - (F.grad(q) * (p - q)).sum(axis=-1)

def MirrorMaptoPostProcessProx(MirrorMap):
    def apply_prox(self, dyn):
       dyn.x = self.grad_conj(dyn.x)
        
    return type(MirrorMap.__name__ + str('_Prox'), 
         (MirrorMap,), 
         dict(
             __call__=apply_prox,
             )
         )


class MirrorMap:
    '''
    Abstract class for mirror maps.
    '''
    def __init__(self, ):
        pass
    
    def __call__(self, theta):
        raise raise_NIE(str(self.__class__), '__call__')
        
    def grad(self, theta):
        raise raise_NIE(str(self.__class__), 'grad')
        
    def grad_conj(self, y):
        raise raise_NIE(str(self.__class__), 'grad_conj')
        
    def hessian(self, theta):
        raise raise_NIE(str(self.__class__), 'hessian')
        
    def Bregman_distance(self, p, q):
        return Bregman_distance(self, p, q)
    
    
class ProjectionMirrorMap(MirrorMap):
    def grad(self, theta):
        return theta


class ProjectionBall(ProjectionMirrorMap):
    def __init__(self, radius=1., center=0.):
        super().__init__()
        self.radius = radius    
        self.center = center
        self.thresh = 1e-5
        
    def __call__(self, theta):
        nx = np.linalg.norm(theta, axis=-1)
        idx = np.where(nx > (self.radius + self.thresh))
        nx = 0.5*nx**2 
        nx[idx] = np.inf
        return nx
    
    def grad_conj(self, y):
        n_y = np.linalg.norm(y - self.center, axis=-1, ord=2, keepdims=True)
        return self.center + (y - self.center) / np.maximum(1, n_y/self.radius)
    

class ProjectionHyperplane(ProjectionMirrorMap):
    def __init__(self, a = None, b = 0):
        super().__init__()
        if a is None:
            a = np.ones((1,1,1))
        self.a = a
        self.norm_a = np.linalg.norm(a, axis=-1)**2
        self.b = b
        
    def grad_conj(self, y):
        return y - ((self.a * y).sum(axis=-1, keepdims=True) - self.b)/self.norm_a * self.a
    
class ProjectionSphere(ProjectionMirrorMap):
    def __init__(self, r=1.):
        super().__init__()
        self.r = r
        
    def grad_conj(self, y):
        return self.r * y/np.linalg.norm(y,axis=-1,keepdims=True)
    
class ProjectionSquare(ProjectionMirrorMap):
    def __init__(self, r=1.):
        self.r = r
    
    def grad_conj(self, y):
        s = y.shape
        y = np.clip(y, a_min=-self.r, a_max=self.r).reshape(-1, s[-1])
        idx = np.argmax(np.abs(y), axis=-1)
        
        y[np.arange(y.shape[0]), idx] = np.sign(y[np.arange(y.shape[0]), idx])
        return y.reshape(s)


    
class ProjectionStiefel(ProjectionMirrorMap):
    def __init__(self, nk=None):
        self.nk = (1,1) if nk is None else nk
    
    def grad_conj(self, y):
        yshape = y.shape
        y = y.reshape((-1,) + self.nk)
        U, _, Vh = np.linalg.svd(y, full_matrices=False)
        return (U@Vh).reshape(yshape)


class LogBarrierBox(MirrorMap):
        
    def __call__(self, theta):
        return np.sum(np.log(1/(1-theta)) + np.log(1/(1+theta)), axis=-1)
    
    def grad(self, theta):
        return 1/(1-theta) - 1/(1+theta)
    
    def grad_conj(self, y):
        return -1/y + 1/y * np.sqrt(1 + y**2)
    
    def hessian(self, theta):
        n,m = theta.shape
        return np.expand_dims(((1/(1-theta))**2 + (1/(1+theta))**2),axis=1)*np.eye(m)
    
class NegativeLogEntropySimplex(MirrorMap):
    
    def grad(self, x):
        x = np.maximum(x, 1e-32)
        return np.log(x) + 1
    
    def grad_conj(self, y):
        return np.exp(y)/(np.exp(y).sum(axis=-1, keepdims=True))
    
    
    
class L2(MirrorMap):
    def __init__(self, lamda=1.0):
        self.lamda = lamda
    
    def __call__(self, theta):
        return self.lamda * 0.5 * np.sum(theta**2,axis=-1)
    
    def grad(self, theta):
        return self.lamda * theta
    
    def grad_conj(self, y):
        return (1/self.lamda) * y
    
    def hessian(self, theta):
        n,m = theta.shape
        return np.expand_dims(np.ones(theta.shape),axis=1)*np.eye(m)
   
    
   
class weighted_L2(MirrorMap):
    def __init__(self, A = None):
        super().__init__()
        if A is None:
            A = np.eye(1)
        self.A = A
    
    def __call__(self,theta):
        return 0.5*theta.T @ self.A @theta
    
    def grad(self, theta):
        return np.reshape(0.5*(self.A + self.A.T)@theta[:,:,None],theta.shape)
    
    def grad_conj(self, y):
        warnings.warn('Not properly implemented', stacklevel=2)
        return y
        #return np.linalg.solve(0.5*(self.A + self.A.T),y.T).T
    
    def hessian(self, theta):
        return np.expand_dims(np.ones(theta.shape),axis=1) * 0.5*(self.A.T+self.A)
    
    
    
class NonsmoothBarrier(MirrorMap):
    def __call__(self, theta):
        return np.sum(np.abs(theta)/(1-np.abs(theta)))
    
    def grad(self, theta):        
        return np.sign(theta)/(1 + np.abs(theta)**2 - 2*np.abs(theta))
    
    def grad_conj(self, y):        
        return np.sign(y) * np.maximum(1-np.sqrt(1/np.abs(y)), 0)
    
    def hessian(self, theta):
        n,m = theta.shape
        
        tmp = (-2*np.abs(theta) + 2)\
            /(-4*np.abs(theta)-4*np.abs(theta)**3+np.abs(theta)**4 + 6*np.abs(theta)**2 + 1)    
        res = np.expand_dims(tmp,axis=1)*np.eye(m)
        
        return res
    
    
class ElasticNet(MirrorMap):
    
    def __init__(self, delta=1.0, lamda=1.0):
        super().__init__()
        self.delta = delta
        self.lamda = lamda
    
    def __call__(self,theta):
        return (1/(2*self.delta))*np.sum(theta**2, axis=-1) + self.lamda*np.sum(np.abs(theta), axis=-1)
    
    def grad(self, theta):
        return (1/(self.delta))*theta + self.lamda * np.sign(theta)
    
    def grad_conj(self, y):
        return self.delta * np.sign(y) * np.maximum((np.abs(y) - self.lamda), 0)
    
    
class Entropy(MirrorMap):
   def __init__(self, scale=1):
       self.scale = scale

   def __call__(self, x):
       return np.sum(x*np.log(1/x), axis = -1)
   
   def grad(self, x):
       EPS = np.finfo(np.float32).eps
       return np.log(1/(x + EPS)) - 1
   
   def grad_conj(self, y):
       return np.exp(-(y + 1))   
    
    
mirror_dict = {
    'ElasticNet': ElasticNet,
    'None': L2, 'L2': L2,
    'ProjectionBall': ProjectionBall,
    'ProjectionHyperplane': ProjectionHyperplane,
    'ProjectionSphere': ProjectionSphere,
    'ProjectionSquare': ProjectionSquare,
    'LogBarrierBox': LogBarrierBox,
    'NonsmoothBarrier': NonsmoothBarrier,
    'weighted_L2': weighted_L2, 
    'Entropy': Entropy,
    'EntropySimplex' : NegativeLogEntropySimplex
    }   


def get_mirror_map_by_name(name, **kwargs):
    if name in mirror_dict.keys():
        return mirror_dict[name](**kwargs)
    else:
        raise ValueError('Unknown mirror map ' + str(name) + '. ' + 
                         ' Please choose from ' + str(mirror_dict.keys()))
        
def get_mirror_map(mm):
    if isinstance(mm, dict):
        return get_mirror_map_by_name(
            mm['name'], 
            **{k:v for (k,v) in mm.items() if not k=='name'}
            )
    elif isinstance(mm, str) or mm is None:
        return get_mirror_map_by_name(str(mm))
    else:
        warnings.warn('MirrorMap did not fit the signature dict or str.' + 
                      'Intepreting the input as a valid mirror map.', 
                      stacklevel=2)
        return mm

#%%
class track_y(track):
    @staticmethod
    def init_history(dyn):
        dyn.history['y'] = []
    
    @staticmethod
    def update(dyn) -> None:
        dyn.history['y'].append(dyn.copy(dyn.y))
    
    
def mirror_step(self,):
    self.compute_consensus() # compute consensus, sets self.energy and self.consensus
    self.drift = self.correction(self.x[self.particle_idx] - self.consensus) # update drift and apply drift correction
    self.y[self.particle_idx] += cbo_update( # perform cbo update step
        self.drift, self.lamda, self.dt, 
        self.sigma, self.noise()
    )
    self.x = self.mirrormap.grad_conj(self.y)
  
def select_mirrormap(self, mirrormap):
    self.mirrormap = get_mirror_map(mirrormap)
    

class MirrorCBO(CBO):
    def __init__(self, f, mirrormap=None, **kwargs):
        super().__init__(f, **kwargs)
        self.select_mirrormap(mirrormap)
        self.y = self.mirrormap.grad(self.copy(self.x))
        
    select_mirrormap = select_mirrormap
    inner_step = mirror_step
        
    known_tracks = {
        'y': track_y,
        **CBO.known_tracks,}