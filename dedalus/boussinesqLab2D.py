"""
Dedalus script for 2D Boussinesq slice using velocity component formulation.

This script uses a Fourier basis in the x direction and a SinCos basis in z.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5
"""

import numpy as np
import pdb #to stop code use pdb.set_trace()

from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)



# Grid parameters and physical constants
Lx, Lz = (.2, .45)
g = 9.81
rho0 = 896.2416 

f_molecular = 1
if f_molecular == 1:
  nu = 1.*10**(-6.)
  kappat = 1.4*10**(-7.)
else: 
  factor = 100.
  nu = 1.*10**(-6.)*factor
  kappat = 1.4*10**(-7.)*factor

gamma = 0.0 
N2 = -g/rho0*gamma


# Create bases and domain
x_basis = de.Fourier('x', 96, interval=(0, Lx), dealias=3/2)
z_basis = de.SinCos('z', 192, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)


# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['phi','b','u','w','s'])
problem.meta['w','b','s']['z']['parity'] = -1
problem.meta['u','phi']['z']['parity'] = 1

problem.parameters['nu'] = nu
problem.parameters['kappat'] = kappat
problem.parameters['N2'] = N2

problem.add_equation("phi = 0", condition="(nx == 0) and (nz == 0)")
problem.add_equation("dx(u) + dz(w) = 0", condition="(nx != 0) or (nz != 0)") 
problem.add_equation("dt(b) - kappat*(dx(dx(b)) + dz(dz(b))) + N2*w    = -(u*dx(b) + w*dz(b))")
problem.add_equation("dt(u) - nu*(dx(dx(u)) + dz(dz(u))) + dx(phi)     = -(u*dx(u) + w*dz(u))")
problem.add_equation("dt(w) - nu*(dx(dx(w)) + dz(dz(w))) + dz(phi) - b = -(u*dx(w) + w*dz(w))")
problem.add_equation("dt(s) - nu*(dx(dx(s)) + dz(dz(s))) = -(u*dx(s) + w*dz(s))")

#Boundary conditions
#Fully periodic domain does not support BCs


# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')


# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

u = solver.state['u']
w = solver.state['w']
phi = solver.state['phi']
b = solver.state['b']
s = solver.state['s']

speed = 0.05
dz_u = 0.01
du = speed*2
alpha = (z-Lz/2.)/dz_u
u['g'] = du/2.*np.tanh(alpha)
w['g'] = 0
b['g'] = 0
phi['g'] = -du/dz_u**2*Lx*np.tanh(alpha)*(1/np.cosh(alpha))**2
#phi['g'] = 0
s['g'] = 0.5*(1+np.tanh(alpha))


# Integration parameters
dt = 0.005
solver.stop_sim_time = 25
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf


# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
snapshots.add_system(solver.state)


# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))


# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
#flow.add_property("0.5(u*u + w*w)", name='KE')


# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
