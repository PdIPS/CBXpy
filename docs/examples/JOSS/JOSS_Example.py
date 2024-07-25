import numpy as np
import cbx as cbx
import cbx.objectives as obj
from cbx.scheduler import multiply
from cbx.plotting import contour_2D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from scipy.interpolate import CubicSpline
from matplotlib.colors import ListedColormap
#%%
kwargs = {'alpha': 1.1,
         'dt': 0.001,
         'sigma': 0.1,
         'lamda': 1.0,
         'd': 2,
         'track_args': {'save_int':1000, 'names':['x']},
         'max_it': 5000,
         'N': 5,
         'M': 1,}

#%% Set seed and define the objective function
np.random.seed(42)
f = obj.Ackley()

#%% Define the initial positions of the particles
x = np.loadtxt('initial.txt',)[None, :,:]
dyn = cbx.dynamics.CBO(f, x=x, **kwargs)
sched = multiply(name='alpha', factor=1.01, maximum=1e3)
x_best = dyn.optimize(sched = sched)

#%% plot particle history
x_hist = np.array(dyn.history['x'])[1:,...]
#idx = [i for i in range(x_hist.shape[0]) if ((i%5==0) or i<4)]
#x_hist = x_hist[idx, ...]
x_min = -4
x_max = 4

plt.close('all')
plt.rcParams.update({
    #"text.usetex": True,
    "font.family": 'STIXGeneral',
    'mathtext.fontset':'stix',
    'font.size':13
})

fig, ax = plt.subplots(1,1)
cf = contour_2D(f, ax=ax, num_pts=1000, 
           x_min=-4, x_max =4., cmap='binary',
           levels=50)
cbar = plt.colorbar(cf)
cbar.set_label('Objective value', rotation=270, labelpad=10)
t = np.linspace(0,1, x_hist.shape[0])
t_eval = np.linspace(0,1, 100 * x_hist.shape[0])
colors = ['xkcd:spruce', 
          'xkcd:seaweed','xkcd:minty green', 
          'xkcd:light seafoam', 'xkcd:grapefruit',
          'xkcd:grapefruit','xkcd:grapefruit', 'xkcd:rose pink']
color_map = ListedColormap(colors)
ax.axis('off')
ax.set_aspect('equal')
ax.set_xlim(x_min,x_max)
ax.set_ylim(x_min,x_max)


for i in range(x_hist.shape[-2]):
    cs_0 = CubicSpline(t, x_hist[:, 0, i, 0])
    cs_1 = CubicSpline(t, x_hist[:, 0, i, 1])
    
    line = Line2D(cs_0(t_eval), cs_1(t_eval), 
                  color='xkcd:spruce',#xkcd:strong blue',
                  alpha=.5,linewidth=3,
                  linestyle='dotted',
                  dash_capstyle='round')
    ax.add_line(line)
    sc_idx = 4
    sc = ax.scatter(x_hist[:sc_idx, 0, i, 0], x_hist[:sc_idx, 0, i, 1], 
               #color=colors[:sc_idx],
               cmap = color_map,
               c = [i for i in range(sc_idx)],
               s=52,zorder=3)
# plt.legend(handles=sc.legend_elements()[0],
#            labels=['t = ' + str(i * kwargs['track_args']['save_int']*dyn.dt) 
#            for i in range(sc_idx)],
#            loc='lower right',
#            title='Particle evolution')
#%%   
save = True
if save:
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0)
    plt.savefig('JOSS.pdf')
