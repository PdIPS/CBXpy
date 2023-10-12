
![cbx](https://github.com/PdIPS/CBXpy/assets/44805883/65e7f1b2-e858-4b8d-af37-8eeaea15214c)

# What is CBXPy?

```CBXPy``` is a python package for consensus-based particle dynamics, focusing on **optimization** and **sampling**. Minimizing a function using CBXPy can be done as follows

```python
   from cbx.dynamics import CBO

   f = lambda x: x[0]**2 + x[1]**2
   dyn = CBO(f, d=2)
   x = dyn.optimize()
```

A Documentation is available at [https://pdips.github.io/CBXpy](https://pdips.github.io/CBXpy)


# Installation



# What is CBX?

For an ensemble of particles $x = (x_1, \ldots, x_N)$ the basic update step reads

$$
x_i \gets x_i - dt\, (x_i - c(x))
$$

CBXPy is a python package implementing consensus based particle schemes. Originally designed for optimization problems

$$
   \min_{x \in \mathbb{R}^n} f(x),
$$

the scheme was introduced as CBO (Consensus Based Optimization). Given an ensemble of points $x = (x_1, \ldots, x_N)$, the update reads

$$
x_i \gets x_i - \lambda\, dt\, (x_i - c(x)) + \sigma\, \sqrt{dt} |x_i - c(x)| \xi_i
$$

where $\xi_i$ are i.i.d. standard normal random variables. The core element is the consensus point

$$c(x) = \frac{\sum_{i=1}^N x_i\, \exp(-\alpha\, f(x_i))}{\sum_{i=1}^N \exp(-\alpha\, f(x_i))}.$$

with a parameter $\alpha>0$. The scheme can be extended to sampling problems  known as CBS, clustering problems and opinion dynamics, which motivates the acronym 
**CBX**, indicating the flexibility of the scheme.



# Usage examples
