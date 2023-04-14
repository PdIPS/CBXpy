import numpy as np
import cbx

#%%
class solver:
    def __init__(self, f, method='cbo', d=2, verbosity=0, T=100, N=50,
                 energy_tol=1e-4,
                 diff_tol=1e-6,
                 **kwargs):
        self.f = f
        self.method = method
        self.d = d
        self.verbosity = verbosity
        self.T = T
        self.N = N
        self.energy_tol = energy_tol

        if method == 'cbo':
            dt = kwargs.get('dt', 0.01)
            sigma = kwargs.get('sigma', 5.0)
            lamda = kwargs.get('lamda', 1.0)
            alpha = kwargs.get('alpha', 10.0)
            x_min = kwargs.get('x_min', -3.)
            x_max = kwargs.get('x_max', 3.)
            x = cbx.utils.init_particles(d = d, N = N, x_min=x_min, x_max = x_max)
            noise = cbx.noise.comp_noise(dt = dt)
            batch_eval = kwargs.get('batch_eval', False)

            self.dyn = cbx.dynamic.CBO(x, f, noise, batch_eval=batch_eval, alpha = alpha, dt = dt, sigma = sigma, lamda = lamda)
            self.scheduler = cbx.scheduler.exponential(self.dyn, r = 1.1)

            def term_crit(t, dyn):
                if t > T:
                    return True
                elif dyn.f_min < energy_tol:
                    return True
                elif dyn.update_diff < diff_tol:
                    return True
                return False
            
            self.term_crit = term_crit


        else:
            raise Warning('Method ' + method + ' not implemented')
        

    def run(self):
        print('-'*20)
        print('Running cbx solver')
        print('Algorithm: ' + self.method)
        print('-'*20)

        t = 0
        it = 0
        while not self.term_crit(t, self.dyn):
            self.dyn.step(t)
            self.scheduler.update()
            t += self.dyn.dt

            if (self.verbosity > 1) and (it%10 == 0):
                print('Time: ' + str(t) + ', best energy: ' + str(self.dyn.f_min))

            it+=1

        print('-'*20)
        print('Finished cbx solver')
        print('Beste energy: ' + str(self.dyn.f_min))
        print('-'*20)

        best_idx = np.argmin(self.dyn.f)
        x_best = self.dyn.x[best_idx,:]
        return x_best

