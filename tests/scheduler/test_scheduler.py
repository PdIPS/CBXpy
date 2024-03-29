import cbx
from cbx.scheduler import multiply, scheduler

def test_multiply_update():
    '''Test if multiply scheduler updates params correctly'''
    dyn = cbx.dynamics.CBO(f=lambda x: x**2, d=1, max_it=1, alpha=1.0, sigma=1.0)
    sched = scheduler([multiply(name='alpha', factor=1.5),
                       multiply(name='sigma', factor=2.0)])
    
    dyn.optimize(sched=sched)
    assert dyn.alpha == 1.5
    assert dyn.sigma == 2.0

def test_multiply_maximum():
    '''Test if multiply scheduler respects maximum'''
    dyn = cbx.dynamics.CBO(f=lambda x: x**2, d=1, max_it=10, alpha=1.0, sigma=1.0)
    sched = scheduler([multiply(name='alpha', factor=1.5, maximum=2.0),
                            multiply(name='sigma', factor=2.0, maximum=4.7)])
    
    dyn.optimize(sched=sched)
    assert dyn.alpha == 2.0
    assert dyn.sigma == 4.7


