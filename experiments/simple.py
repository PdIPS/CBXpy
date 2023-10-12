from cbx.dynamics import CBO

f = lambda x: x.sum(axis=-1)**2
dyn = CBO(f, d=2, M=2)
x = dyn.optimize()
print(x)