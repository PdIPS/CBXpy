import numpy as np

class Termination:
    def __init__(self, checks: list, 
                 M: int = 1, 
                 term_on_all: bool = True, 
                 verbosity: int = 0):
        self.checks = checks
        self.term_reason = [None for _ in range(M)]
        self.M = M
        self.verbosity = verbosity
        self.term_on_all = term_on_all
        self.all_check = np.zeros(M, dtype=bool)

    def terminate(self, dyn):
        """
        Terminate the process and return a boolean value indicating if for each run the termination criterion was met.

        Parameters:
            verbosity (int): The level of verbosity for printing information. Default is 0.

        Returns:
            bool: True if all checks passed, False otherwise.
        """
        loc_check = np.zeros((self.M, len(self.checks)), dtype=bool)
        for i,check in enumerate(self.checks):
            loc_check[:,i] = check(dyn)
            
        all_check = np.sum(loc_check, axis=1)
            
        for j in range(self.M):
            if all_check[j] and not self.all_check[j]:
                self.term_reason[j] = np.where(loc_check[j,:])[0]
        self.all_check = all_check

        return self.checks_to_bool()
    
    def checks_to_bool(self,):
        if self.term_on_all:
            return self.check_on_all()
        else:
            return self.check_on_any()
    
    def check_on_all(self,):
        term = False
        if np.all(self.all_check):
            for j in range(self.M):
                if self.verbosity > 0:
                    print('Run ' + str(j) + ' returning on checks: ')
                    for k in self.term_reason[j]:
                        print(self.checks[k].__name__)
            term = True
        return term
    
    def check_on_any(self,):
        term = False
        if np.any(self.all_check):
            for j in range(self.M):
                if self.verbosity > 0 and self.all_check[j]:
                    print('Run ' + str(j) + ' returning on checks: ')
                    for k in self.term_reason[j]:
                        print(self.checks[k].__name__)
            term = True
        return term

def check_energy(dyn):
    """
    Check if the energy is below a certain tolerance.

    Returns:
        bool: True if the energy is below the tolerance, False otherwise.
    """
    return dyn.f_min < dyn.energy_tol
    
def check_diff_tol(dyn):
    """
    Checks if the update difference is less than the difference tolerance.

    Returns:
        bool: True if the update difference is less than the difference tolerance, False otherwise.
    """
    return dyn.update_diff < dyn.diff_tol
    
def check_max_eval(dyn):
    """
    Check if the number of function evaluations is greater than or equal to the maximum number of evaluations.

    Returns:
        bool: True if the number of function evaluations is greater than or equal to the maximum number of evaluations, False otherwise.
    """
    return dyn.num_f_eval >= dyn.max_eval
    
def check_max_it(dyn):
    """
    Checks if the current value of `self.it` is greater than or equal to the value of `self.max_it`.

    Returns:
        bool: True if `self.it` is greater than or equal to `self.max_it`, False otherwise.
    """
    return dyn.it >= dyn.max_it

def check_max_time(dyn):
    """
    Checks if the current value of `self.it` is greater than or equal to the value of `self.max_it`.

    Returns:
        bool: True if `self.it` is greater than or equal to `self.max_it`, False otherwise.
    """
    return dyn.t >= dyn.max_time