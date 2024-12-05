import numpy as np
from warnings import warn


#%%
class track:
    """
    Base class for tracking of variables in the history dictionary of given dynamics.
    """

    @staticmethod
    def init_history(dyn) -> None:
        """
        Initializes the value to be tracked in the history dictionary of the given dyn object.
        
        Parameters
        ----------
        dyn : object
            The object to track in the history dictionary.
            
        Returns
        -------
        None
        """
        pass

    @staticmethod
    def update(dyn) -> None:
        """
        Updates the value to be tracked in the history dictionary of the given dyn object.
        
        Parameters
        ----------
        dyn : object
            The object to track in the history dictionary.
            
        Returns
        -------
        None
        """
        pass

#%%
class default_track(track):
    def __init__(self, name):
        self.name = str(name)
        self.tracking = True
        
    def init_history(self, dyn) -> None:
        dyn.history[self.name] = []

    
    def update(self, dyn) -> None:
        if self.tracking:
            if hasattr(dyn, self.name):
                dyn.history[self.name].append(
                    dyn.copy(getattr(dyn, self.name))
                )
            else:
                warn('The tracker tried to track the variable ' + 
                      self.name + 
                      ' which is not an attribute of the given dynamic. ' + 
                      'The varibale will not be tracked!',
                      stacklevel=2)
                self.tracking = False
    
#%%
class track_x(track):
    """
    Class for tracking of variable 'x' in the history dictionary.
    """
    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['x'] = []
        dyn.history['x'].append(dyn.x)
    
    @staticmethod
    def update(dyn) -> None:
        """
        Update the history of the 'x' variable by copying the current particles to the next time step.

        Parameters
        ----------
            dyn : object
                The object to track in the history dictionary.

        Returns
        -------
            None
        """
        dyn.history['x'].append(dyn.copy(dyn.x))
        
        
class track_update_norm(track):
    """
    Class for tracking the 'update_norm' entry in the history.

    """

    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['update_norm'] = []
    
    @staticmethod
    def update(dyn) -> None:
        """
        Updates the 'update_norm' entry in the 'history' dictionary with the 'update_diff' value.

        Parameters:
            None

        Returns:
            None
        """
        dyn.history['update_norm'].append(dyn.update_diff)
     

class track_energy(track):
    """
    Class for tracking the 'energy' entry in the history.
    """

    @staticmethod
    def init_history(dyn):
        dyn.history['energy'] = []

    @staticmethod
    def update(dyn) -> None:
        dyn.history['energy'].append(np.copy(dyn.energy))# always assumes energy is numpy


class track_consensus(track):
    """
    Class for tracking the 'consensus' entry in the dynamic.
    """

    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['consensus'] = []
    @staticmethod
    def update(dyn) -> None:
        dyn.history['consensus'].append(dyn.copy(dyn.consensus))
        
class track_drift_mean(track):
    """
    Class for tracking the 'drift_mean' entry in the history.
    """

    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['drift_mean'] = []
    @staticmethod
    def update(dyn) -> None:
        dyn.history['drift_mean'].append(np.mean(np.abs(dyn.drift), axis=(-2,-1)))
        
class track_drift(track):
    """
    Class for tracking the 'drift' entry in the history.
    """

    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['drift'] = []
        dyn.history['particle_idx'] = []
    
    @staticmethod
    def update(dyn) -> None:        
        dyn.history['drift'].append(dyn.drift)
        dyn.history['particle_idx'].append(dyn.particle_idx)