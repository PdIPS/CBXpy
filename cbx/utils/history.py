import numpy as np

class track_x:
    """
    Class for tracking of variable 'x' in the history dictionary.

    Parameters:
        None

    Returns:
        None
    """
    @staticmethod
    def init_history(dyn):
        dyn.history['x'] = []
        dyn.history['x'].append(dyn.x)
    
    @staticmethod
    def update(dyn) -> None:
        """
        Update the history of the 'x' variable by copying the current particles to the next time step.

        Parameters:
            None

        Returns:
            None
        """
        dyn.history['x'].append(dyn.copy_particles(dyn.x))
        
        
class track_update_norm:
    """
    Class for tracking the 'update_norm' entry in the history.

    Returns:
        None
    """

    @staticmethod
    def init_history(dyn):
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
     

class track_energy:
    """
    Class for tracking the 'energy' entry in the history.

    Returns:
        None
    """

    @staticmethod
    def init_history(dyn):
        dyn.history['energy'] = []

    @staticmethod
    def update(dyn) -> None:
        dyn.history['energy'].append(dyn.best_cur_energy) 


class track_consensus:
    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['consensus'] = []
    @staticmethod
    def update(dyn) -> None:
        dyn.history['consensus'].append(dyn.copy_particles(dyn.consensus))
        
class track_drift_mean:
    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['drift_mean'] = []
    @staticmethod
    def update(dyn) -> None:
        dyn.history['drift_mean'].append(np.mean(np.abs(dyn.drift), axis=(-2,-1)))
        
class track_drift:
    @staticmethod
    def init_history(dyn) -> None:
        dyn.history['drift'] = []
        dyn.history['particle_idx'] = []
    
    @staticmethod
    def update(dyn) -> None:        
        dyn.history['drift'].append(dyn.drift)
        dyn.history['particle_idx'].append(dyn.particle_idx)