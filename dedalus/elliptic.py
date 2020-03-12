import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, 2))
y_basis = de.Fourier('y', 256, interval=(0, 2))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
x = domain.grid(0)
y = domain.grid(1)


# Poisson equation
problem = de.LBVP(domain, variables=['u','uy'])
problem.add_equation("dx(dx(u)) + dy(uy) = 0", condition = "(nx != 0) or (ny != 0)")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("u = 0",  condition = "(nx == 0) and (ny == 0)")


# Build solver
solver = problem.build_solver()
solver.stop_iteration = 1

#Initial conditions
u = solver.state['u']
uy = solver.state['uy']
u['g'] = 10.
uy['g'] = 100.


#Results = solver.evaluator.add_file_handler('results2/state', sim_dt=60, max_writes=60, mode='append')
#Results.add_system(solver.state)


#Run
solver.solve()

# Plot solution
u = solver.state['u']
u.require_grid_space()
plot_tools.plot_bot_2d(u)
plt.savefig('poisson.png')
