import pyswarms as ps
import numpy as np
import math
options = {"c1": 0.5, "c2": 0.3, "w":0.9}
x_max= [1,1,1,1,1,1]
x_min= [0,0,0,0,0,0]

my_bounds = (x_min, x_max)

bounds = my_bounds

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds= my_bounds)

def endurance(arr: np.array):
    x = arr[0]
    y = arr[1]
    z = arr[2]
    v = arr[3]
    u = arr[4]
    w = arr[5]

    return -(math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u) +math.cos(v*w))

def f(x):
    n_particles = x.shape[0]
    j = [endurance(x[i]) for i in range(n_particles)]
    return np.array(j)

optimizer.optimize(f, iters=1000)






