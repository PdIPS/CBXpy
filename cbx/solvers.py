import numpy as np
import sys
import cbx

#%%
class solver:
    def __init__(
        self, f, 
        f_dim = '1D', d=2, 
        method='cbo',  verbosity: int =0, 
        T: float = 100., N: int = 50,
        dt: float = 0.1, r: float = 1.1,
        alpha: float = 1.0, sigma: float =1.0,
        lamda: float = 1.0,
        energy_tol: float = 1e-4,
        diff_tol: float = 1e-6,
        num_runs: int = 1,
        max_eval: int = float('inf'),
        M: int = 1, parallel: bool = False,
        num_cores: int = 1,
        **kwargs):


        self.f = f
        self.num_f_eval = 0
        self.f_min = float('inf')
        self.method = method
        self.d = d
        self.verbosity = verbosity
        self.T = T
        self.N = N
        self.M = M
        self.energy_tol = energy_tol
        self.parallel = parallel
        self.num_cores = num_cores
        self.num_runs = num_runs
        self.x_min = kwargs.get('x_min', -1.)
        self.x_max = kwargs.get('x_max', 1.)
        batch_size = kwargs.get('batch_size', None)
        self.noise = cbx.noise.comp_noise(dt = dt)
        self.r = r

        if method == 'cbo':
            self.DYN = cbx.dynamic.CBO
            self.DYN_args = {
                    'f_dim': f_dim,
                    'alpha': alpha,
                    'dt': dt,
                    'sigma': sigma,
                    'lamda': lamda,
                    'batch_size': batch_size,
                    'T': T,
                    'max_eval': max_eval,
                    'energy_tol': energy_tol,
                    'diff_tol': diff_tol,
                }
        else:
            raise Warning('Method ' + method + ' not implemented')
        

    def solve(self):
        print('-'*20, flush = True)
        print('Running cbx solver', flush = True)
        print('Algorithm: ' + self.method, flush = True)
        print('-'*20, flush = True)

        for i in range(self.num_runs):
            x = cbx.utils.init_particles(shape=(self.M,self.N,self.d), x_min = self.x_min, x_max = self.x_max)
            dyn = self.DYN(x, self.f, self.noise, **self.DYN_args)
            scheduler = cbx.scheduler.exponential(dyn, r=self.r)
            solver = cbx_solver(
                dyn, scheduler, idx = i, 
                verbosity=self.verbosity)
            x_best_new, f_min_new = solver.run()
            self.num_f_eval += np.sum(solver.dyn.num_f_eval)

            f_min_new = np.min(f_min_new)
            if f_min_new < self.f_min:
                self.f_min = f_min_new
                f_min_idx = np.argmin(f_min_new)
                x_best = x_best_new[f_min_idx, :]
                best_run_idx = i

        
        print('='*20)
        print('Run: ' + str(best_run_idx) + ' returned the best particle with f_min = ' + str(self.f_min))
        print('Number of function evaluations: ' + str(self.num_f_eval))
        print('='*20)
        return x_best


class cbx_solver:
    def __init__(self, dyn, scheduler, 
                 idx: int = 0, 
                 verbosity: int = 0,
                 print_int: int = 100):
        self.dyn = dyn
        self.scheduler = scheduler
        self.verbosity = verbosity
        self.it = 0
        self.idx = idx
        self.print_int = print_int
    
    def run(self):
        print('.'*20)
        print('Starting run ' + str(self.idx))
        print('.'*20)
        while not self.dyn.terminate():
            self.dyn.step()
            self.scheduler.update()

            if (self.verbosity > 1) and (self.dyn.it % self.print_int == 0):
                print('Time: ' + "{:.3f}".format(self.dyn.t) + ', best energy: ' + str(self.dyn.f_min))
                print('Number of function evaluations: ' + str(self.dyn.num_f_eval))

                if self.verbosity > 2:
                    print('Current alpha: ' + str(self.dyn.alpha))

        print('-'*20)
        print('Finished cbx run ' + str(self.idx))
        print('Best energy: ' + str(self.dyn.f_min))
        print('-'*20)

        return self.dyn.best_particle(), self.dyn.f_min