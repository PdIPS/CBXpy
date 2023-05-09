import numpy as np
import sys
import cbx
import multiprocessing as mp
from contextlib import closing

#%%
class solver:
    def __init__(self, f, method='cbo', d=2, verbosity: int =0, T: float = 100., N: int = 50,
                 energy_tol: float = 1e-4,
                 diff_tol: float = 1e-6,
                 M: int = 1, parallel: bool = False,
                 num_cores: int = 1,
                 **kwargs):
        self.f = f
        self.method = method
        self.d = d
        self.verbosity = verbosity
        self.T = T
        self.N = N
        self.energy_tol = energy_tol
        self.parallel = parallel
        self.num_cores = num_cores

        if method == 'cbo':
            dt = kwargs.get('dt', 0.01)
            sigma = kwargs.get('sigma', 8.0)
            lamda = kwargs.get('lamda', 1.0)
            alpha = kwargs.get('alpha', 30.0)
            x_min = kwargs.get('x_min', -1.)
            x_max = kwargs.get('x_max', 1.)
            max_eval = kwargs.get('max_eval', float('inf'))
            batch_size = kwargs.get('batch_size', None)
            batch_eval = kwargs.get('batch_eval', False)
            noise = cbx.noise.comp_noise(dt = dt)
            
            self.runs = []

            for i in range(M):
                x = cbx.utils.init_particles(d = d, N = N, x_min=x_min, x_max = x_max)
                dyn = cbx.dynamic.CBO(
                        x, f, noise, batch_eval=batch_eval, 
                        alpha = alpha, dt = dt, sigma = sigma, 
                        lamda = lamda, max_eval = max_eval,
                        batch_size=batch_size)
                self.runs.append(cbx_run(dyn, cbx.scheduler.exponential(dyn, r = 1.1), i, 
                                         verbosity=verbosity,
                                         energy_tol=energy_tol, diff_tol=diff_tol,
                                         T=T))

        else:
            raise Warning('Method ' + method + ' not implemented')
        

    def solve(self):
        print('-'*20, flush = True)
        print('Running cbx solver', flush = True)
        print('Algorithm: ' + self.method, flush = True)
        print('-'*20, flush = True)

        if self.parallel:
            self.runs = self.parallel_run()
        else:
            for run in self.runs:
                single_cbx_run(run, verbosity=self.verbosity)

        f_min = np.float('inf')
        for run in self.runs:
            if run.dyn.f_min < f_min:
                f_min = run.dyn.f_min
                x_best = run.x_best
                idx_best = run.idx
        
        print('='*20)
        print('Run: ' + str(idx_best) + ' returned the best particle with f_min = ' + str(f_min))
        print('Number of function evaluations: ' + str(self.runs[idx_best].dyn.num_f_eval))
        print('='*20)
        return x_best


    def parallel_run(self,):
        num_cores = min(mp.cpu_count(), self.num_cores)
        pool = mp.Pool(num_cores)

        with closing(pool):
            p = list(pool.imap_unordered(single_cbx_run, self.runs))
            #print(p)
            pool.close()
            pool.join()
        return p


class cbx_run:
    def __init__(self, dyn, scheduler, idx, 
                 verbosity: int = 0,
                 diff_tol: float = 1e-6,
                 energy_tol: float = 1e-4,
                 T: float = 100.):
        self.dyn = dyn
        self.scheduler = scheduler
        self.verbosity = verbosity
        self.it = 0
        self.idx = idx
        self.diff_tol = diff_tol
        self.energy_tol = energy_tol
        self.T = T
    
def single_cbx_run(run, verbosity=0):
    print('.'*20)
    print('Starting run ' + str(run.idx))
    print('.'*20)
    while not run.dyn.terminate():
        run.dyn.step()
        run.scheduler.update()

        if (verbosity > 1):
            print('Time: ' + "{:.3f}".format(run.dyn.t) + ', best energy: ' + str(run.dyn.f_min))

            if verbosity > 2:
                print('Current alpha: ' + str(run.dyn.alpha))

    print('-'*20)
    print('Finished cbx run ' + str(run.idx))
    print('Best energy: ' + str(run.dyn.f_min))
    print('-'*20)

    best_idx = np.argmin(run.dyn.energy)
    x_best = run.dyn.x[best_idx,:]
    run.x_best = x_best
    return run