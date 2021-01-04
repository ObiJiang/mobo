import math
import numpy as np
from GP import MultiObjsGaussianProcess_Ind

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import sobol_seq

from tqdm import tqdm

np.random.seed(0)

""" Functions to optimize (max) """
input_dim = 2
nb_init_sampels = 1
nb_iters = 100
debug_plot = False

from benchmarks import branin,Currin
functions=[branin,Currin]
nb_objs = len(functions)

""" Define Random grid, input/design space and optimum """
# generate random sobol grids
grid = sobol_seq.i4_sobol_generate(input_dim,1000,np.random.randint(0,100))
# in finite domain, it will also be the whole design space
design_space = grid
nb_points = design_space.shape[0]

# outputs (real y)
y_real = np.zeros((design_space.shape[0], nb_objs))
for j in range(nb_objs):
    for i, x in enumerate(design_space):
        y_real[i][j] = functions[j](x,input_dim)

def sample(X, functions, noise=0.1):
    assert(len(X.shape) == 2)
    nb_points = X.shape[0]
    input_dim = X.shape[-1]
    nb_objs = len(functions)
    noise = np.random.normal(loc=0, scale=noise, size=(nb_points, nb_objs))

    y_real = np.zeros((nb_points, nb_objs))
    for j in range(nb_objs):
        for i, x in enumerate(X):
            y_real[i][j] = functions[j](x,input_dim)
    
    return y_real + noise

# define lambdas
lambdas = np.array([[1,0], [0,1]])
Z = np.zeros((lambdas.shape[0], nb_objs))
for l in range(lambdas.shape[0]):
    Z[l,:] = y_real[np.argmax(y_real @ lambdas[l:l+1,:].T),:]

# too slow
# if debug_plot:
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(design_space[:,0], design_space[:,1], y_real[:,0:1], rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
#     ax.set_title('first function')
#     plt.show()
#     print("hwllp")

""" Declare and initialize GPs """
multi_gps = MultiObjsGaussianProcess_Ind(input_dim, nb_objs)

if debug_plot:
    plt.scatter(grid[:,0], grid[:,1])
    plt.show()

for k in range(nb_init_sampels):
    exist=True
    while exist:
        design_index = np.random.randint(0, grid.shape[0])
        x_rand=list(grid[design_index : (design_index + 1), :][0])
        exist = multi_gps.check_if_x_exists(x_rand, 0)

    # y = [ functions[i](x_rand,input_dim) for i in range(nb_objs) ]
    y = sample(np.array([x_rand]), functions)
    multi_gps.addSamples(x_rand, y[0])

multi_gps.fitModel()

def cover_cost(lambdas, z, y):
    assert(lambdas.shape[0] == z.shape[0])
    assert(len(y.shape) == 2)
    nb_ls = lambdas.shape[0]
    costs_per_l = np.zeros(nb_ls)
    for i, (l, z) in enumerate(zip(lambdas, Z)):
        costs_per_l[i] = np.min((z-y) @ l)

    return np.max(costs_per_l)

def new_cumulative_regret(lambdas, z, y):
    nb_ls = lambdas.shape[0]
    costs_per_l = np.zeros(nb_ls)
    for i, (l, z) in enumerate(zip(lambdas, Z)):
        costs_per_l[i] = (z-y) @ l
    
    return np.min(costs_per_l)
        
def acq_functions(mo_models, x, lambdas, t):
    beta_t = 2 * np.log(nb_objs * nb_points * (t+1)**2 * 16.44)
    mean, std = multi_gps.getPrediction(design_space)
    ucb = mean + beta_t * std

    ucb_scalars = []
    for l in lambdas:
        ucb_scalars.append(np.expand_dims(ucb @ l, axis=1))
    
    max_ucb_scalars = np.max(np.concatenate(ucb_scalars, axis=1),axis=1)
    max_x_ind = np.argmax(max_ucb_scalars)

    return x[max_x_ind:max_x_ind+1, :]


""" Main Loops """
cumulative_regret = 0
X_T = []
f_T = []
regrets = []
for t in tqdm(range(nb_iters)):
    # select the next point to sample
    x_t = acq_functions(multi_gps, design_space, lambdas, t)

    # samples (+ noise)
    y_t = sample(x_t, functions)
    f_t = sample(x_t, functions, noise=0)
    f_T.append(f_t)

    # calculate regret
    cumulative_regret += new_cumulative_regret(lambdas, Z, f_t)
    cover_regret = cover_cost(lambdas, Z, np.vstack(f_T))
    regret = cumulative_regret/(t+1) + cover_regret
    regrets.append(regret)

    # add samples
    multi_gps.addSamples(x_t[0], y_t[0])

    # fit model
    multi_gps.fitModel()

fig = plt.figure()
plt.plot(range(nb_iters), regrets)
fig.suptitle('Regret Plot', fontsize=15)
plt.xlabel('t', fontsize=15)
plt.ylabel('Regret', fontsize=15)
plt.savefig("results/regret_plot")

    
    
