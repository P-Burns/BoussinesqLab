from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, \
    cos, sin, exp, pi, SpatialCoordinate, Constant, Function, as_vector, DirichletBC, \
    FunctionSpace, VectorFunctionSpace, interpolate
#from firedrake.petsc import PETSc
#from firedrake import parameters
#from pyop2.profiling import timed_stage
import numpy as np
import sympy as sp
from sympy.stats import Normal
import sys
import matplotlib.pyplot as plt
#import CosineSineTransforms as cst
import pdb as pdb
import itertools as itertools
from pyop2.mpi import MPI


#PETSc.Log.begin()
#parameters["pyop2_options"]["lazy_evaluation"] = False


# Programme control:
CheckPoint = True
Pickup = False
#ParkRun = 14
#ParkRun = 18
ParkRun = -1

ICsNon0 = 1
ICsSimpleWave = 0
ICsGaussian = 0
ICsRandom = 1
Interpolate = 0
FilterField = 0

AddNonRandomForce = 0
AddWaveForce = 0
AddDedalusForce = 0
AddRandomForce = 0

Inviscid = 0
MolecularDiffusion = 1
EddyDiffusion = 0
ScaleDiffusion = 1

AdjustmentCase = 1

#Set some time control options:
#dt = 1./20
#dt = 0.01
#dt = 0.005
#dt = 0.0075
dt = 0.001
#dt = 0.00075

if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 5*60


##############################################################################
# set up mesh
##############################################################################
# Construct 1d periodic base mesh for idealised lab experiment of Park et al. (1994)
factor = 1
columns = 80
columns = columns*factor
L = 0.2
m = PeriodicIntervalMesh(columns, L)

# build 2D mesh by extruding the base mesh
nlayers = 180
nlayers = nlayers*factor
H = 0.45  # Height position of the model top
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
x = SpatialCoordinate(mesh)


##############################################################################
# set up all the other things that state requires
##############################################################################

# list of prognostic fieldnames
# this is passed to state and used to construct a dictionary,
# state.field_dict so that we can access fields by name
# u is the 2D velocity
# p is the pressure
# b is the buoyancy
fieldlist = ['u', 'p', 'b']


# class containing timestepping parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py
subcycles = 4
timestepping = TimesteppingParameters(dt=dt*subcycles)
#timestepping = TimesteppingParameters(dt=dt, adaptive=True, CourantLimit=0.20, maxDt=0.1, maxFracIncreaseDt=0.001)

# class containing output parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py

#Get vector of coordinates:
V_DG0 = FunctionSpace(mesh, "DG", 0)
W = VectorFunctionSpace(mesh, V_DG0.ufl_element())
X = interpolate(mesh.coordinates, W)
#points_x = X.dat.data_ro[:,0]
#points_x = [points_x[int(columns/2.)]]

points_x = np.array([0.10125])
points_z = X.dat.data_ro[:,1].flatten()
points_z = np.sort(np.unique(points_z)).tolist()

#points_x = np.array([0.10125])
#points_z = np.linspace(0, H, nlayers+1)

points = np.array([p for p in itertools.product(points_x, points_z)])

#print("Rank: ", MPI.COMM_WORLD.rank, " points: ", points)

w2f_points= 1
if w2f_points == 1:
    fnm = './points.txt' 
    np.savetxt(fnm,points)

#dtOutput = 0.001
#dtOutput = .1
dtOutput = .1
dumpfreq = int(dtOutput/(dt*subcycles))

output = OutputParameters(dirname='tmp', dumpfreq=dumpfreq, dumplist=['u','b'], 
perturbation_fields=['b'], checkpoint=CheckPoint, point_data=[('b_gradient', points)]  )
#output = OutputParameters(dirname='tmp', dumpfreq_method = "time", dumpfreq=dumpfreq, dumplist=['u','b'], 
#perturbation_fields=['b'], checkpoint=False, timestepping=True)
#point_data=[('b_gradient', points)] 

# class containing physical parameters
# all values not explicitly set here use the default values provided
# and documented in configuration.py

# Physical parameters adjusted for idealised lab experiment of Park et al. (1994):
if ParkRun == 14: N2=0.35
if ParkRun == 18: N2=3.83
if ParkRun == -1:
    #N2 = 0.25
    #N2 = 1
    #N2 = 2.25
    N2 = 4
    #N2 = 6.25
    #N2 = 9
    #N2 = 12.25
    #N2 = 16
    #N2 = 20.25
    #N2 = 25

parameters = CompressibleParameters(N=np.sqrt(N2))


# class for diagnostics
# fields passed to this class will have basic diagnostics computed
# (eg min, max, l2 norm) and these will be output as a json file
diagnostics = Diagnostics(*fieldlist)

# list of diagnostic fields, each defined in a class in diagnostics.py
diagnostic_fields = [CourantNumber(), Gradient("b")]


# set up state, passing in the mesh, information on the required finite element
# function spaces, z, k, and the classes above
state = State(mesh, vertical_degree=1, horizontal_degree=1,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)


##############################################################################
# Initial conditions
##############################################################################

# set up functions on the spaces constructed by state
u0 = state.fields("u")
p0 = state.fields("p")
b0 = state.fields("b")

# first set up the background buoyancy profile
# z.grad(bref) = N**2
# the following is symbolic algebra, using the default buoyancy frequency
# from the parameters class. x[1]=z and comes from x=SpatialCoordinate(mesh)
N = parameters.N
bref = N**2*(x[1]-H)
# interpolate the expression to the function
Vb = b0.function_space()
b_b = Function(Vb).interpolate(bref)

# Define bouyancy perturbation to represent background soup of internal waves in idealised lab scenario of Park et al.
# The reference density was found by estimating drho/dz from Figures of Park et al. (1994), converting to SI units,
# and then using the N^2 value.
g = parameters.g

dgamma = 100./3
dz_b = 2./100
a0 = 100.
z_a = H/2
rhoprime13 = dgamma*z_a + a0*dz_b + dgamma/2*dz_b

#Make comparison to Run 18 by only changing background gradient:
drho0_dz13 = -122.09
N2_18 = 3.83
drho0_dz_18 = -425.9
rho0 = -g/N2_18*drho0_dz_18
rhoprime = rhoprime13 * drho0_dz_18/drho0_dz13
bprime = -g/rho0*rhoprime


if ICsNon0 == 1:
    if ICsSimpleWave == 1:
        #b_pert = bprime/2.*sin(k1*x[0]) + bprime/2.*sin(m1*x[1])
        #b_pert = bprime/2. * sin(k1*x[0]+m1*x[1])
        b_pert = bprime/2. * sin(m1*x[1])
    if ICsGaussian == 1:
        sigma = 0.01
        b_pert = bprime*exp( -( x[1] - H/2 )**2 / (2*sigma**2) )
        #options that did not work:
        #b_pert = sp.Piecewise( (0, x[1] < H/2-0.01), (0, x[1] > H/2+0.01), (A_z1, H/2-0.01 >= x[1] <= H/2+0.01, True) )
        #b_pert = sp.integrate( A_z1 * DiracDelta(x[1]-H/2), (x[1],0,H) )
    if ICsRandom == 1:
        #r = Function(b0.function_space()).assign(Constant(0.0))
        #r.dat.data[:] += np.random.uniform(low=-1., high=1., size=r.dof_dset.size)
        #b_pert = r*bprime*20

        #Get vector of coordinates:
        V_DG0 = FunctionSpace(mesh, "DG", 0)
        W = VectorFunctionSpace(mesh, V_DG0.ufl_element())
        X = interpolate(mesh.coordinates, W)

        dx = L/columns
        dz = H/nlayers

        if Interpolate == 1:
            #Read in the Aegir random field:  
            if factor == 1: RandomSample = np.loadtxt('/home/ubuntu/BoussinesqLab/RandomSample_080_180.txt')
            if factor == 2: RandomSample = np.loadtxt('/home/ubuntu/BoussinesqLab/RandomSample_160_360.txt')

            #get Aegir coefficients and re-normalise:
            fhat_Aegir = cst.FFT_FST(columns, nlayers, RandomSample)/(np.float_(columns))

            kk = np.fft.fftfreq(columns,L/columns)*2.*np.pi # wave numbers used in Aegir
            kk_cosine = np.arange((nlayers))*np.pi/H   # wave numbers used in Aegir

            def InterpolateFromAegir(dx,dz,Nx,Nz,X, kk,kk_cosine, uHatNumpy):
                """
                Nx and Nz are the size of the Aegir arrays
                X is the grid for Gusto
                kk and kk_cosine are from Aegir
                uHatNumpy is the spectral coefficents as computed with Aegir
                """
                fOutput = np.zeros((Nx,Nz))*1j

                for (x, z) in X:
                    for i in range(Nx):
                        for j in range(1,Nz-1):
                            idxX = int(x/dx)
                            idxZ = int(z/dz)
                            fOutput[idxX,idxZ] += np.sin(kk_cosine[j]*z)*np.exp(-1j*kk[i]*x)*uHatNumpy[i,j]
                return fOutput.real

            RandomSample_Gusto = InterpolateFromAegir(dx,dz,columns,nlayers,X.dat.data_ro, kk,kk_cosine, fhat_Aegir)
            #plt.contourf(RandomSample_Gusto.transpose())
            #plt.show()

            fnm = '/home/ubuntu/BoussinesqLab/RandomSample_080_180_Gusto.txt'
            np.savetxt(fnm,RandomSample_Gusto)

        #Read in the random field:  
        if factor == 1: RandomSample = np.loadtxt('/home/ubuntu/BoussinesqLab/RandomSample_080_180_Gusto.txt')
        if factor == 2: RandomSample = np.loadtxt('/home/ubuntu/BoussinesqLab/RandomSample_160_360_Gusto.txt')
        RandomSample = RandomSample/np.max(RandomSample)

        RandomSample = RandomSample*bprime*5

        def ExternalData(dx,dz,x,z, data):
            return data[int(x/dx),int(z/dz)]

        def mydata(X):
            list_of_output_values = []
            for (x, z) in X:
                list_of_output_values.append(ExternalData(dx,dz,x,z, RandomSample))
            return list_of_output_values

        b_pert_dg0 = Function(V_DG0)
        b_pert_dg0.dat.data[:] = mydata(X.dat.data_ro)
        b_pert = Function(Vb)
        b_pert.interpolate(b_pert_dg0)
        
else: b_pert = 0

# interpolate the expression to the function:
b0.interpolate(b_b + b_pert)

#Balance equations:
incompressible_hydrostatic_balance(state, b_b, p0, top=False)

# pass these initial conditions to the state.initialise method
state.initialise([("u", u0), ("p", p0), ("b", b0)])

# set the background buoyancy
state.set_reference_profiles([("b", b_b)])


##############################################################################
# Set up advection schemes
##############################################################################
# advection_dict is a dictionary containing field_name: advection class
ueqn = EulerPoincare(state, u0.function_space())
supg = True
if supg:
    beqn = SUPGAdvection(state, Vb,
                         equation_form="advective")
else:
    beqn = EmbeddedDGAdvection(state, Vb,
                               equation_form="advective")
advected_fields = []
#advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
#advected_fields.append(("b", SSPRK3(state, b0, beqn)))
advected_fields.append(("u", SSPRK3(state, u0, ueqn, subcycles=subcycles)))
advected_fields.append(("b", SSPRK3(state, b0, beqn, subcycles=subcycles)))


##############################################################################
# Set up linear solver for the timestepping scheme
##############################################################################
linear_solver = IncompressibleSolver(state)


##############################################################################
# Set up forcing
#############################################################################

if AddNonRandomForce == 1:
    if AddWaveForce == 1:
        #These are the wavelengths observed in the lab experiments:
        lmda_x1 = 2.0/100
        lmda_z1 = 2.0/100

        #Domain is periodic in x so we get as close to the 
        #observations are possible:
        k_int = 10
        k1 = 2*np.pi*k_int/L
        #Domain is not periodic in z so we can exactly mimic the 
        #observations:
        m1 = 2*pi/lmda_z1
            
        omega = 2*pi*N

        A_f = bprime/2.

        f_ux = -m1/k1*A_f * sin(x[0]*k1 + x[1]*m1 - omega*state.t)
        f_uz = A_f * sin(x[0]*k1 + x[1]*m1 - omega*state.t)

    if AddDedalusForce == 1:
        k_int = 10
        k1 = 2*np.pi*k_int/L
        m_int = 22
        m1 = 2*np.pi*m_int/H
           
        omega = np.sqrt(N2)*(2*np.pi)

        A_f = bprime/2.
        f_uz = A_f * cos(k1*x[0]) * sin(m1*x[1]) * sin(omega*state.t)
        f_ux = -A_f * m1/k1 * sin(k1*x[0]) * cos(m1*x[1]) * sin(omega*state.t)

    f_u = as_vector([f_ux,f_uz])
    forcing = IncompressibleForcing(state, extra_terms=f_u)

if AddRandomForce == 1:
    forcing = RandomIncompressibleForcing(state)

if (AddNonRandomForce == 0) and (AddRandomForce == 0):
    forcing = IncompressibleForcing(state)


##############################################################################
#Set up diffusion scheme and any desired BCs
##############################################################################
if Inviscid == 0:
    # mu is a numerical parameter
    # kappa is the diffusion constant for each variable
    # Note that molecular diffusion coefficients were taken from Lautrup, 2005:
    if MolecularDiffusion == 1:
        kappa_u = 1.*10**(-6.)
        kappa_b = 1.4*10**(-7.)
    if EddyDiffusion == 1:
        kappa_u = 10.**(-2.)
        kappa_b = 10.**(-2.)
    if ScaleDiffusion == 1:
        DiffScaleFact_u = 100.
        DiffScaleFact_b = 100.
        kappa_u = kappa_u * DiffScaleFact_u
        kappa_b = kappa_b * DiffScaleFact_b

    Vu = u0.function_space()
    Vb = state.spaces("HDiv_v")
    delta = L/columns		#Grid resolution (same in both directions).

    if AdjustmentCase == 1:
        #fctr = 1./2
        fctr = np.sqrt(1./2)
        #fctr = np.sqrt(7./8)
        #fctr = 2

        BCz0 = -(fctr*N)**2*H

        #BCzH = 0
        BCzH = -(fctr*N)**2*H
    else:
        BCz0 = -N**2*H
        BCzH = 0

    bcs_u = [DirichletBC(Vu, 0.0, "bottom"), DirichletBC(Vu, 0.0, "top")]
    bcs_b = [DirichletBC(Vb, BCz0, "bottom"), DirichletBC(Vb, BCzH, "top")]

    diffused_fields = []
    diffused_fields.append(("u", InteriorPenalty(state, Vu, kappa=kappa_u,
                                           mu=Constant(10./delta), bcs=bcs_u )))
    diffused_fields.append(("b", InteriorPenalty(state, Vb, kappa=kappa_b,
                                           mu=Constant(10./delta), bcs=bcs_b )))


##############################################################################
# build time stepper
##############################################################################


if Inviscid == 1:
    stepper = CrankNicolson(state, advected_fields, linear_solver, forcing)
else:
    stepper = CrankNicolson(state, advected_fields, linear_solver, forcing, diffused_fields)


##############################################################################
# Run!
##############################################################################
#PETSc.Log.begin()
#with timed_stage("Timings"):
#    stepper.run(t=0, tmax=tmax,pickup=Pickup)
stepper.run(t=0, tmax=tmax,pickup=Pickup)
#stepper.run(t=48,tmax=tmax,pickup=CheckPoint)
