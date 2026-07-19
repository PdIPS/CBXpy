import warnings
import numpy as np
from .cbo import CBO, cbo_update
from ..scheduler import scheduler
#%%
class CBS(CBO):
    r"""Consensus-Based Sampling (CBS) class.

    Implements CBS as described in [1]_. Two discretisation schemes are available:

    * ``scheme='EM'`` — Euler-Maruyama (inherited CBO step):

      .. math::
          X^{k+1} = X^k - dt\,(X^k - c_\alpha) + \text{noise}

    * ``scheme='exponential'`` — exponential integrator (exact linear solve):

      .. math::
          X^{k+1} = c_\alpha + e^{-dt}(X^k - c_\alpha) + \text{noise}

    In both cases the noise uses the covariance noise model with the matching factor
    (:math:`\sqrt{2\,dt/\lambda}` for EM and :math:`\sqrt{(1-e^{-2dt})/\lambda}` for the
    exponential integrator), with :math:`\lambda = 1 + \alpha`.

    Parameters
    ----------
    f : callable
        The objective function.
    mode : str, optional
        Passed to :class:`covariance_noise <cbx.noise.covariance_noise>`. Default: ``'sampling'``.
    noise : str, optional
        Noise model. Default: ``'covariance'``.
    scheme : str, optional
        Discretisation scheme: ``'EM'`` or ``'exponential'``. Default: ``'EM'``.

    References
    ----------
    .. [1] Carrillo, J. A., Hoffmann, F., Stuart, A. M., & Vaes, U. (2022).
        Consensus-based sampling. Studies in Applied Mathematics, 148(3), 1069-1140.
    """

    def __init__(self, f, mode='sampling', noise='covariance',
                 scheme='EM',
                 M=1,
                 track_args=None,
                 **kwargs):
        track_args = track_args if track_args is not None else {'names': []}
        super().__init__(f, track_args=track_args, noise=noise, M=M, **kwargs)
        self.sigma = 1.

        if self.batched:
            raise NotImplementedError('Batched mode not implemented for CBS!')
        if self.x.ndim > 3:
            raise NotImplementedError('Multi dimensional domains not implemented for CBS! The particle should have the dimension M x N x d, where d is an integer!')

        if noise not in ['covariance', 'sampling']:
            warnings.warn('For CBS usually covariance or sampling noise is used!', stacklevel=2)

        if scheme not in ('EM', 'exponential'):
            raise ValueError(f"scheme must be 'EM' or 'exponential', got '{scheme}'")

        self.scheme = scheme
        self.noise_callable.mode = mode
        self.noise_callable.scheme = scheme

    def inner_step(self):
        self.compute_consensus()
        self.drift = self.x[self.particle_idx] - self.consensus
        self.s = self.sigma * self.noise()

        if self.scheme == 'exponential':
            self.x[self.particle_idx] = (
                self.consensus +
                np.exp(-self.dt) * self.drift +
                self.s
            )
        else:  # EM
            self.x[self.particle_idx] += cbo_update(
                self.correction(self.drift), self.lamda, self.dt, 1.0, self.s
            )
        
    def run(self, sched = 'default'):
            if self.verbosity > 0:
                print('.'*20)
                print('Starting Run with dynamic: ' + self.__class__.__name__)
                print('.'*20)

            if sched is None:
                sched = scheduler([])
            elif sched == 'default':
                sched = self.default_sched()
            else:
                if not isinstance(sched, scheduler):
                    raise RuntimeError('Unknonw scheduler specified!')

            while not self.terminate():
                self.step()
                sched.update(self)
                
    def optimize(self, sched = 'default'):
        self.run(sched=sched)
        
    def default_sched(self,):
        return scheduler([])
        
    def process_particles(self,):
        pass