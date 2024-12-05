import timeit
import cbx
from cbx.dynamics import CBO
import numpy as np
from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures


f = cbx.objectives.Quadratic()
x = np.random.uniform(-3,3, (10,100,200))
rep = 10


class MultithreadedRNG:
    def __init__(self, n, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s)
                                   for s in seq.spawn(threads)]

        self.n = n
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(n)
        self.step = np.ceil(n / threads).astype(np.int_)

    def fill(self):
        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last])

        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    self.values,
                    i * self.step,
                    (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        
    def __call__(self, size=None):
        self.fill()
        return self.values.reshape(size)
        
    

    def __del__(self):
        self.executor.shutdown(False)



rng = np.random.default_rng(12345)
mrng = MultithreadedRNG(x.size, seed=12345)
samplers = [np.random.normal, np.random.standard_normal, rng.standard_normal, mrng]

for sampler in samplers:
    dyn = CBO(f, x=x, sampler=sampler)
    T = timeit.Timer(dyn.step)

    r = T.repeat(rep, number=50)
    best = sum(r)/rep
    print(str(sampler) + ' Mean ' + str(rep) + ': ' +str(best)[:6] + 's')