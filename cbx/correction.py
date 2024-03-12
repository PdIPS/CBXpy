from typing import Any, Callable
from numpy.typing import ArrayLike
import numpy as np

def get_correction(name, **kwargs):
    if name == 'no_correction':
        return no_correction()
    elif name == 'heavi_side':
        return heavi_side_correction()
    elif name == 'heavi_side_reg':
        eps = kwargs.get('eps', 1e-3)
        return heavi_side_reg_correction(eps=eps)
    else:
        raise ValueError('Unknown correction ' + str(name))

class correction:
    def __init__(self):
        pass

class no_correction(correction):
    """
    The class if no correction is specified. Is equal to the identity.

    Parameters:
        x: The input value.

    Returns:
        The input value without any correction.
    """

    def __call__(self, dyn, x):
        return self.correct(x)
    
    def correct(self, x:Any) -> Any:
        """
        Return the identity
        
        Parameters
        ----------
        x : Any
            The input array.
        
        Returns
        -------
        x: Any
            The input array.
        """

        return x
    
class heavi_side_correction(correction):
    """
    Calculate the Heaviside correction for the given input.

    """

    def __call__(self, dyn, x : ArrayLike) -> ArrayLike:
        """
        Calculate the Heaviside correction for the given input.
        
        Parameters
        ----------
        dyn : CBXDynamic
            The CBXDynamic object.
        x : ndarray
            The input array.

        Returns
        -------
        ndarray
            The Heaviside correction value.

        .. note::
            This function evaluates the objective function on the consensus, therfore the number of function evaluations is increased by the consensus size.
        """
        dyn.num_f_eval += dyn.consensus.shape[0] # update number of function evaluations
        return self.correct(x, dyn.f, dyn.energy, dyn.consensus)

    def correct(self, x:ArrayLike, f: Callable, energy: ArrayLike, consensus: ArrayLike) -> ArrayLike:
        z = energy - f(consensus)
        return x * np.where(z > 0, 1,0)[...,None]

class heavi_side_reg_correction(heavi_side_correction):
    """
    Calculate the Heaviside regularized correction.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def correct(self, x:ArrayLike, f: Callable, energy: ArrayLike, consensus: ArrayLike) -> ArrayLike:
        z = energy - f(consensus)
        return x * (0.5 + 0.5 * np.tanh(z/self.eps))[...,None]
    

