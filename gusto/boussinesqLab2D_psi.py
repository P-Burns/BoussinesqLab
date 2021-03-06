"""
Dedalus script for 2D Boussinesq slice using the streamfunction/vorticity formulation.

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
import matplotlib.pyplot as plt
from dedalus.core.operators import GeneralFunction
import os as os
import randomPhaseICs as ics
import CosineSineTransforms as cst


#Program control:
Restart 		= 0
#ProblemType 		= "BarotropicKH"
ProblemType 		= "Layers"
#ParkRun 		= 14   #for Park et al run 14
#ParkRun 		= 18   #for Park et al run 18
ParkRun 		= -1
scalePert		= 0
#N2			= 0.09
#N2			= 0.25
#N2			= 1
N2			= 2.25
#N2			= 4
#N2			= 6.25
#N2			= 7.5625
#N2			= 9
#N2			= 10.5625
#N2			= 12.25
#N2			= 14.0625
#N2			= 16
#N2			= 20.25
#N2			= 25

Inviscid          	= 0
ImplicitDiffusion	= 1
MolecularDiffusion 	= 1
ScaleDiffusion 		= 1

ICsRandomPert 		= 1
ReadICs 		= 1
Interpolate		= 0
MeshTest		= 0
ICsWaves 		= 0
ICsTestModulation	= 0

AddForce 		= 0
ForceFullDomain 	= 1
ForceSingleColumn 	= 0

PassiveTracer 		= 0
compute_p		= 0

CoordinateRotation	= 0
nvars			= 2

Linear			= 0

domain3D		= 0

w2f_grid 		= 0
w2f_state 		= 1
w2f_SinglePoint 	= 0
w2f_dt		 	= 1
w2f_energy		= 0


# Create bases and domain
Lx, Lz 	= (.2, .45)
Nx 	= 80
Nz 	= 180
#factor	= 1./4
#factor	= 1./2
factor	= 1
#factor	= 2
#factor	= 4
Nx 	= int(Nx*factor)
Nz 	= int(Nz*factor)
if factor == 1./4: Nz += 1

x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.SinCos('z', Nz, interval=(0, Lz), dealias=3/2)

if domain3D == 1: 
    Ny = 1
    Ly = 1 
    y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)

domain 	= de.Domain([x_basis, z_basis], grid_dtype=np.float64)
#Use COMM_SELF so keep calculations independent between processes:
#domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)
x = domain.grid(0)
z = domain.grid(1)

#print(x)
#print(z)
#print(z[0,1]-z[0,0])
#pdb.set_trace()

if w2f_grid == 1:
    dir_grid = './'
    fnm_gridx = dir_grid + 'XGridDedalus.txt'
    fnm_gridz = dir_grid + 'ZGridDedalus.txt'
    np.savetxt(fnm_gridx,x)
    np.savetxt(fnm_gridz,z)
    pdb.set_trace()


#Set physical constants
g = 9.81
ct = 2.0*10**(-4.)
cs = 7.6*10**(-4.) 

if Inviscid == 0:
    if MolecularDiffusion == 1:
        nu = 1.*10**(-6.)
        kappat = 1.4*10**(-7.)
        kappas = 1.4*10**(-7.)
    if ScaleDiffusion == 1:
        ScaleFact_T = 100
        ScaleFact_nu = 100
        ScaleFact_S = 100
        nu = nu*ScaleFact_nu
        kappat = kappat*ScaleFact_T
        kappas = kappas*ScaleFact_S

if ProblemType == "BarotropicKH":
    bt = 0.
    bs = 0. 

if ProblemType == "Layers":
    #Typical density perturbations for Park et al lab run 13.
    #Scaled off plot:
    dgamma = 100./3
    dz_b = 2./100
    a0 = 100.
    z_a = Lz/2.
    rhoprime13 = dgamma*z_a + a0*dz_b + dgamma/2*dz_b

    #We now scale run 13 perturbations using the background 
    #density field:
    drho0_dz13 = -122.09

    if ParkRun == 14:
        N2 = 0.35                       #Taken from Park et al paper
        drho0_dz = -31.976              #Scaled off Park et al plot
        rho0 = -g/N2*drho0_dz
        rhoprime = rhoprime13 * drho0_dz/drho0_dz13
        Spert0 = rhoprime * 1./(rho0*cs)
    if ParkRun == 18:
        N2 = 3.83                       #Taken from Park et al paper
        drho0_dz = -425.9               #Scaled off Park et al plot
        rho0 = -g/N2*drho0_dz
        rhoprime = rhoprime13 * drho0_dz/drho0_dz13
        Spert0 = rhoprime * 1./(rho0*cs)
    if ParkRun < 0:                     #For arbitrary N^2
        if scalePert == 1:
            #Set ref density constant:
            N2_18 = 3.83
            drho0_dz_18 = -425.9
            rho0 = -g/N2_18*drho0_dz_18

            #assume perturbations are constant fraction of background density field:
            const = rhoprime13/drho0_dz13

            #Find initial background density field given some 
            #user defined N^2 and the constant ref density: 
            drho0_dz = -N2*rho0/g
            #scale perturbations:
            rhoprime = drho0_dz*const
            Spert0 = rhoprime * 1./(rho0*cs)
        
        if scalePert == 0:
            #Make comparison to Run 18 by only changing background gradient:
            N2_18 = 3.83
            drho0_dz_18 = -425.9
            rho0 = -g/N2_18*drho0_dz_18
            rhoprime = rhoprime13 * drho0_dz_18/drho0_dz13
            Spert0 = rhoprime * 1./(rho0*cs)

            drho0_dz = -N2/g*rho0
            #print(drho0_dz) 
            #pdb.set_trace()

    bs = -1./(rho0*cs)*drho0_dz
    bt = 0.


#Define governing equations:
if compute_p==0:
    variables=['psi','T','S']
    problem = de.IVP(domain, variables=variables)
    problem.meta['psi','T','S']['z']['parity'] = -1
else: 
    variables=['psi','T','S','p']
    problem = de.IVP(domain, variables=variables)
    problem.meta['psi','T','S']['z']['parity'] = -1
    problem.meta['p']['z']['parity'] = 1

if PassiveTracer == 1:
    s = domain.new_field()
    s.meta['z']['parity'] = -1

if AddForce == 1:
    F = domain.new_field()
    F.meta['z']['parity'] = -1

    if ForceFullDomain == 1:
        #k1 = 2*np.pi/dz_b - this is from the physics - 
        #we need to get as close to this as possible in the numerical problem.
        k_int = 10/2
        k1 = 2*np.pi*k_int/Lx
        #m_int = 22/2
        m_int = 22/2
        m1 = 2*np.pi*m_int/Lz

        #kmag2 = k1**2 + m1**2
        #N2 = -g*(ct*bt-cs*bs)
        #omega = sqrt(k1**2/kmag2*N2) 

        #omega = np.sqrt(N2)*(2*np.pi)/100
        omega = np.sqrt(N2)*(2*np.pi)/10
        #omega = np.sqrt(N2)*(2*np.pi)
        #omega = 0.3*(2*np.pi) 

        #A_f = Spert0/10000.
        #A_f = Spert0/1000.
        A_f = Spert0/100.
        #A_f = Spert0/10.
   
        #F['g'] = A_f*np.sin(m1*z) + 0*x
        #F['g'] = A_f*np.sin(m1*z)*np.cos(omega*time.time()) 
        F['g'] = (m1**2/k1+k1)*A_f*np.sin(k1*x)*np.sin(m1*z)*np.sin(omega*time.time())
      
        #F['g'] = A_f*np.sin(k1*x-omega*time.time()) 		# invalid - incorrect parity 
        #F['g'] = A_f*np.sin(m1*z-omega*time.time())		# invalid - mixing parities
        #F['g'] = Spert0/2.*np.sin(k1*x+m1*z-omega*time.time()) 	# invalid - mixing parities

    if ForceSingleColumn == 1:
       
        #mask = domain.new_field()
        #mask.meta['z']['parity'] = 1
        #mask['g']=0
        c1 = 1.
        c2 = Lz/2.
        c3 = Lz/10. 
        #mask['g'][Nx-1,:] = c1 * np.exp( -(z-c2)**2/(2*c3**2) )
        #mask['g'][Nx-1,:] = 1
        #print(mask['g'][Nx-1,:])
        #pdb.set_trace()

        k_int = 10/2
        k1 = 2*np.pi*k_int/Lx
        m_int = 22/2
        m1 = 2*np.pi*m_int/Lz

        omega = np.sqrt(N2)*(2*np.pi)
 
        A_f = Spert0/100.
     
        #F['g'] = mask['g'] * m1**2*A_f*np.sin(m1*z)*np.sin(omega*time.time())*Lx
        mask = c1*np.exp(-(z-c2)**2/(2*c3**2))
        #print(mask)
        F['g'] = 0
        #F['g'][Nx-1,:] = m1**2*A_f*np.sin(m1*z)*np.sin(omega*time.time())*Lx * mask
        F['g'][0,:] = (m1**2/k1+k1)*A_f*np.sin(k1*x[0])*np.sin(m1*z)*np.sin(omega*time.time())

        #print(F['g'][Nx-2,:])
        #pdb.set_trace()


if CoordinateRotation == 1:

    #Define subclass of GeneralFunction and add meta_parity definition:
    class GF_parity(GeneralFunction):
        #n.b. GeneralFunction(self, domain, layout, func, args=[], kw={}, out=None,)

        #Add meta_parity module to new class.
        #This is the definition for adding prognostic fields, which is 
        #essentially what the linear algebra does.
        def meta_parity(self, axis):
            # Parities must match
            parity0 = self.args[0].meta[axis]['parity']
            parity1 = self.args[1].meta[axis]['parity']
            if parity0 != parity1:
                raise UndefinedParityError("Cannot add fields of different parities.")
            else:
                return parity0

    #Define coordinate rotation Python function:
    def CR(*args):

        t = args[2].value
        out = args[3]

        #Read in eigenvectors computed by PostProcessDedalus.py: 
        if N2 == 0.25: RunName = 'StateN2_00_25'
        if N2 == 1: RunName = 'StateN2_01'
        if N2 == 2.25: RunName = 'StateN2_02_25'
        if N2 == 4: RunName = 'StateN2_04'
        if N2 == 6.25: RunName = 'StateN2_06_25'
        if N2 == 7.5625: RunName = 'StateN2_07_5625'
        if N2 == 9: RunName = 'StateN2_09'
        if N2 == 10.5625: RunName = 'StateN2_10_5625'
        if N2 == 12.25: RunName = 'StateN2_12_25'
        if N2 == 14.0625: RunName = 'StateN2_14_0625'
        if N2 == 16: RunName = 'StateN2_16'
        if N2 == 20.25: RunName = 'StateN2_20_25'
        if N2 == 25: RunName = 'StateN2_25'
        dir_ivec = './Results/' + RunName + '/NaturalBasis/'

        ivec_1 = np.zeros((Nx,Nz,nvars))
        ivec1 = np.zeros((Nx,Nz,nvars))
        for ll in range(0,nvars):
            fnm_ivec_1 = dir_ivec + 'ivec_1_' + str(ll+1) + '.txt'
            fnm_ivec1 = dir_ivec + 'ivec1_' + str(ll+1) + '.txt'
            ivec_1[:,:,ll] = np.loadtxt(fnm_ivec_1)
            ivec1[:,:,ll] = np.loadtxt(fnm_ivec1)

        #Initialise Dedalus objects for calculations:
        psi_r = domain.new_field()
        S_r = domain.new_field()
        psi_r.meta['z']['parity'] = -1
        S_r.meta['z']['parity'] = -1

        #Get wavenumbers for calculations on distributed grid:
        kk = domain.elements(0).flatten()
        kk_cosine = domain.elements(1).flatten()

        g = problem.parameters['g']
        cs = problem.parameters['cs']
        bs = problem.parameters['bs']
        ct = problem.parameters['ct']
        bt = problem.parameters['bt']

        c2 = np.sqrt(cs/bs)

        #Make coordinate rotation looping through wavenumber space:
        for jj in range(0,len(kk_cosine)):
            for ii in range(0,len(kk)):

                kIdx = np.where(x_basis.wavenumbers == kk[ii])[0][0]
                nIdx = np.where(z_basis.wavenumbers == kk_cosine[jj])[0][0]

                r_1 = ivec_1[kIdx,nIdx,:]
                r1 = ivec1[kIdx,nIdx,:]

                k = kk[ii]
                n = kk_cosine[jj]
                kvec = np.array([k,n])
                kmag = np.linalg.norm(kvec)

                if kmag != 0:

                    psi_hat = psi['c'][ii,jj]*kmag
                    S_hat = S['c'][ii,jj]*np.sqrt(g)*c2
                    #psi_hat = psi['c'][ii,jj]
                    #S_hat = S['c'][ii,jj]
                    f_hat_vec = np.array([psi_hat, S_hat])

                    EigenVecsM = np.array([r_1,r1]).transpose()
                    EigenVecsM_inv = np.linalg.inv(EigenVecsM)
 
                    c3 = abs(k)/kmag*np.sqrt(-(ct*bt-cs*bs))          #N.B. omega = c3*sqrt(g)

                    diagonalM = np.zeros((2,2))*1j
                    omega_1 = -c3*np.sqrt(g)*1j
                    omega1 = c3*np.sqrt(g)*1j
                    diagonalM[0,0] = np.exp(-omega_1*t)
                    diagonalM[1,1] = np.exp(-omega1*t)

                    MatrixExp = np.mat(EigenVecsM)*np.mat(diagonalM)*np.mat(EigenVecsM_inv)
                    fnc_r = np.mat(MatrixExp)*np.mat(f_hat_vec).T

                    psi_r['c'][ii,jj] = fnc_r[0,0]
                    S_r['c'][ii,jj] = fnc_r[1,0]
                else:
                    psi_r['c'][ii,jj] = psi['c'][ii,jj]
                    S_r['c'][ii,jj] = S['c'][ii,jj]

        if out == 0: return S_r['c']
        if out == 1: return psi_r['c']

    #Define a function which will return subclass GF_parity, 
    #which takes function CR as an argument:
    def GF_cr(*args, domain=domain, F=CR):
        return GF_parity(domain, layout='c', func=F, args=args)

    #Add function GF_cr to list of symbols used in equations.
    #N.b. we don't have to give function arguments here:
    de.operators.parseables['GF_cr'] = GF_cr


#Parameters
problem.parameters['g'] = g
if Inviscid == 0:
    problem.parameters['nu'] = nu
    problem.parameters['kappat'] = kappat
    problem.parameters['kappas'] = kappas
problem.parameters['ct'] = ct
problem.parameters['cs'] = cs
problem.parameters['bt'] = bt
problem.parameters['bs'] = bs
problem.parameters['Lz'] = Lz
if AddForce == 1: problem.parameters['F'] = F
if compute_p == 1: problem.parameters['rho0'] = rho0

#Substitutions:
problem.substitutions['L(psi)'] = "d(psi,x=2) + d(psi,z=2)"
problem.substitutions['u'] = " dz(psi)"
problem.substitutions['w'] = "-dx(psi)"

#Momentum equation:
LHS_1 = "dt(L(psi))-g*(cs*dx(S)-ct*dx(T))"
RHS_1 = ""
if Inviscid == 0:
    if ImplicitDiffusion == 1: LHS_1 += "-nu*L(L(psi))"
    else: RHS_1 += "nu*L(L(psi))"
if AddForce == 1: RHS_1 += "+F"
if Linear == 0: RHS_1 += "-u*dx(L(psi))-w*dz(L(psi))"
if Linear == 1 and (Inviscid == 1 or ImplicitDiffusion == 1): RHS_1 = "0"
momentum_eq = LHS_1 + " = " + RHS_1
print(momentum_eq)
problem.add_equation(momentum_eq, condition = "(nx != 0) or (nz != 0)")
problem.add_equation("psi = 0", condition = "(nx == 0) and (nz == 0)")

#Temperature equation:
LHS_2 = "dt(T)+bt*dx(psi)"
RHS_2 = ""
if Inviscid == 0:
    if ImplicitDiffusion == 1: LHS_2 += "-kappat*(d(T,x=2)+d(T,z=2))"
    else: RHS_2 += "kappat*(d(T,x=2)+d(T,z=2))"
if Linear == 0: RHS_2 += "-dz(psi)*dx(T)+dx(psi)*dz(T)"
if Linear == 1 and (Inviscid == 1 or ImplicitDiffusion == 1): RHS_2 = "0"
T_eq = LHS_2 + " = " + RHS_2
print(T_eq)
problem.add_equation(T_eq)

#Salinity equation:
LHS_3 = "dt(S)+bs*dx(psi)"
RHS_3 = ""
if Inviscid == 0:
    if ImplicitDiffusion == 1: LHS_3 += "-kappas*(d(S,x=2)+d(S,z=2))"
    else: RHS_3 += "kappas*(d(S,x=2)+d(S,z=2))"
if Linear == 0: RHS_3 += "-dz(psi)*dx(S)+dx(psi)*dz(S)"
if Linear == 1 and (Inviscid == 1 or ImplicitDiffusion == 1): RHS_3 = "0"
S_eq = LHS_3 + " = " + RHS_3
print(S_eq)
problem.add_equation(S_eq)

#For computing pressure (derived from other prognostics) by 
#solving a Poisson equation. This is required for energy calculations:
if compute_p == 1:
    LHS_p = "1/rho0*L(p) + cs*g*dz(S)"
    RHS_p = ""
    if AddForce == 1: RHS_p += "+F_x"
    if Linear == 0: RHS_p += "-( dx(u*dx(u)+w*dz(u)) + dz(u*dx(w)+w*dz(w)) )"
    if Linear == 1 and (Inviscid == 1 or ImplicitDiffusion == 1): RHS_p = "0"
    p_eq = LHS_p + " = " + RHS_p
    print(p_eq)
    problem.add_equation(p_eq, condition = "(nx != 0) or (nz != 0)")
    problem.add_equation("p = 0", condition = "(nx == 0) and (nz == 0)")

#Passive tracer equation:
if PassiveTracer == 1:
    LHS_tr = "dt(s)"
    LHS_tr = ""
    if Inviscid == 0:
        if ImplicitDiffusion == 1: LHS_tr += "-nu*(d(s,x=2)+d(s,z=2))"
        else: RHS_tr += "nu*(d(s,x=2)+d(s,z=2))"
    if Linear == 0: RHS_tr += "-dz(psi)*dx(s)+dx(psi)*dz(s)"
    if Linear == 1 and (Inviscid == 1 or ImplicitDiffusion == 1): RHS_tr = "0"
    tracer_eq = LHS_tr + " = " + RHS_tr
    problem.add_equation(tracer_eq)


#Boundary conditions
#Fully periodic domain does not support BCs


# Build solver
#solver = problem.build_solver(de.timesteppers.RK222)
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')


# Initial conditions
psi = solver.state['psi']
T = solver.state['T']
S = solver.state['S']
if PassiveTracer == 1: s = solver.state['s']

if ProblemType == "BarotropicKH":
    speed = 0.01
    dz_u = 0.001
    alpha = (z-Lz/2.)/dz_u

    DefinePsi = 1
    if DefinePsi == 1:
        du = speed*2
        #u['g'] = du/2.*np.tanh(alpha)
        psi['g'] = du*dz_u/2.*( np.log(2) - Lz/(2.*dz_u) + np.log(np.cosh(alpha)) )

        AddPert = 1
        if AddPert == 1:
            PsiAbsMax = np.max(np.abs(psi['g']))
            A1 = 0.05
            psi_prime_max = PsiAbsMax*A1
            A2 = 0.2
            wavelength = Lx*A2  #wave perturbation is 4 cm long
            k1 = 2*np.pi/wavelength
            psi['g'][:,int(Nz/2)] = psi['g'][:,int(Nz/2)] + psi_prime_max*np.sin(k1*x[:,0])
            #pdb.set_trace()

    else:
        #Read in spectral coefficents for Psi:
        dir_psi_hat = './'
        fnm_psi_hat = dir_psi_hat + 'Psi_hat_0.txt'
        Psi_hat = np.loadtxt(fnm_psi_hat).view(complex)
        psi['c'] = Psi_hat[0:int(Nx/2),:]
        #plt.contourf(psi['g'].transpose(),50)
        #plt.colorbar()
        #plt.show()
        #pdb.set_trace()

    w['g'] = 0
    T['g'] = 0
    S['g'] = 0
    if PassiveTracer == 1: s['g'] = 0.5*(1+np.tanh(alpha))

if ProblemType == "Layers":
    psi['g'] = 0
    T['g'] = 0
    S['g'] = 0
    if PassiveTracer == 1: s['g'] = 0

    if ICsRandomPert == 1:
        if ReadICs == 0:
            kkz = z_basis.wavenumbers

            kkx = x_basis.wavenumbers
            #Dedalus drops the negative wavenumbers for the Fourier basis, 
            #so here we add them back due to our smoothing method:
            kkx_neg = np.flipud(kkx[1:]*(-1))
            kkx = np.append(kkx,kkx_neg)
            #check:
            #plt.plot(kkx)
            #plt.show()
            #print(kkx)

            #Parameters for our smoothing method:
            kx_0 = 6
            kz_0 = 14
            m = 25

            #Define initial random field:
            tmp = np.random.uniform(low=-1,high=1,size=(Nx+1,Nz))
            #n.b. an extra point in x was added so that we can follow 
            #the Dedalus method of dropping the last point in x.
 
            #Give field correct parity for chosen basis.
            #We make the field periodic in x:
            tmp_flipx = np.flipud(tmp)
            tmp_even_x = (tmp + tmp_flipx)
            tmp_even_x_flipz = np.fliplr(tmp_even_x)
            tmp_even_odd = (tmp_even_x - tmp_even_x_flipz)

            #Define quasi-random field in the Dedalus virtual env:
            tmp0 = domain.new_field()
            tmp0.meta['z']['parity'] = -1
            tmp0['g'] = tmp_even_odd[0:Nx,:]	#here we drop the last point in x
       
            #Define coefficients for negative wavenumbers in x using
            #Fourier transform symmetry relation for Real functions: 
            tmp0Hat = np.zeros((Nx,Nz), dtype=complex)
            #Add coefficients for positive wavenumbers given by Dedalus:
            tmp0Hat[0:int(Nx/2.),:] = tmp0['c']
            for j in range(0,Nz):
                for i in range(0,int(Nx/2.)):
    		    #Apply symmetry property of real-valued functions:
                    tmp0Hat[int(Nx/2.)-i,j] = np.conjugate(tmp0['c'][i,j])

            #Pass Fourier coefficients to smoothing method:
            tmp1Hat = ics.randomPhaseICs(tmp0Hat,Lx,Lz,kkx,kkz,kx_0,kz_0,m,plot=False)
        
            RandomSample = domain.new_field()
            RandomSample.meta['z']['parity'] = -1
            RandomSample['c'] = tmp1Hat[int(Nx/2.):Nx,:]

            #plt.figure()
            #plt.contourf(RandomSample['g'])
            #plt.show()
            pdb.set_trace()

        #Read in random sample
        if ReadICs == 1:

            basePath = "/gpfs/ts0/home/pb412/BoussinesqLab/"

            if Interpolate == 1:

                #Read in Aegir ICs:
                if factor == 1: RandomSample = np.loadtxt('./RandomSample_080_180.txt')
                if factor == 2: RandomSample = np.loadtxt('./RandomSample_160_360.txt')

                #check symmetry:
                #print(RandomSample[:,1])
                #print(RandomSample[0,:])
                #pdb.set_trace()

                #get Aegir coefficients and re-normalise:
                fhat_Aegir = cst.FFT_FST(Nx, Nz, RandomSample)/(np.float_(Nx))

                kk = np.fft.fftfreq(Nx,Lx/Nx)*2.*np.pi # wave numbers used in Aegir
                kk_cosine = np.arange((Nz))*np.pi/Lz   # wave numbers used in Aegir

                def InterpolateFromAegir(Nx, Nz, xgridAlt, zgridAlt, kk, kk_cosine, uHatNumpy):
                    """
                    Nx and Nz are the size of the Aegir arrays
                    xgrid and zgrid are the grids from Dedalus and can be any size
                    kk and kk_cosine are from Aegir
                    uHatNumpy is the spectral coefficents as computed with Aegir
                    """
                    fOutput = np.zeros((len(xgridAlt),len(zgridAlt)))*1j
                    for ix in range(len(xgridAlt)): # Loops over the xgrid
                        for iz in range(len(zgridAlt)):# Loops over the zgrid
                            fOutput[ix,iz] = 0.
                            for i in range(Nx):
                                for j in range(1,Nz-1):
                                    fOutput[ix, iz] += np.sin(kk_cosine[j]*zgridAlt[iz])*np.exp(-1j*kk[i]*xgridAlt[ix])*uHatNumpy[i,j]
                    return fOutput.real

                RandomSample_Dedalus = InterpolateFromAegir(Nx, Nz, x[:,0], z[0,:], kk, kk_cosine, fhat_Aegir)
          
                if factor == 1: fname = './RandomSample_080_180_Dedalus.txt'
                if factor == 2: fname = './RandomSample_160_360_Dedalus.txt'
                np.savetxt(fname, RandomSample_Dedalus) 
                #pdb.set_trace()

            if MeshTest == 1:
                if factor == 1./4: RandomSample = np.loadtxt('./RandomSample_020_046.txt')
                if factor == 1./2: RandomSample = np.loadtxt('./RandomSample_040_090.txt')
                if factor == 1: RandomSample = np.loadtxt('./RandomSample_080_180.txt')
                if factor == 2: RandomSample = np.loadtxt('./RandomSample_160_360.txt')
                if factor == 4: RandomSample = np.loadtxt('./RandomSample_320_720.txt')
            else:
                if factor == 1: RandomSample = np.loadtxt(basePath + './RandomSample_080_180_Dedalus.txt')
                if factor == 2: RandomSample = np.loadtxt(basePath + './RandomSample_160_360_Dedalus.txt')
            RandomSample = RandomSample/np.max(RandomSample)
        
            #check symmetry:
            #plt.plot(RandomSample[:,1])
            #plt.figure()
            #plt.plot(RandomSample[0,:])
            #plt.show()
            #pdb.set_trace()

            S0 = RandomSample*Spert0*5
            slices = domain.dist.grid_layout.slices(scales=1) 
            S0 = S0[slices]
            S['g'] = S0

    if ICsWaves == 1: 
        #Impose a wave field
        N_waves_x = 2
        N_waves_z = 2
        factor = 2.
        S_pert = np.sin(N_waves_x*2*np.pi/Lx*x)*np.sin(N_waves_z*2*np.pi/Lz*z)*Spert0/factor
        S['g'] = S['g'] + S_pert 

    #This is not required as pressure is not a prognostic variable but rather derived from the 
    #other prognostics.
    #This is a method for initialising pressure by solving the Poisson equation in spectral space.
    #compute_p0=0
    #if compute_p0 == 1:
    #    kk = domain.elements(0).flatten()
    #    kk_cosine = domain.elements(1).flatten()
    #    #kk_cosine = z_basis.wavenumbers
    #    #kk = x_basis.wavenumbers
    #
    #    g = problem.parameters['g']
    #    cs = problem.parameters['cs']
    #    rho0 = problem.parameters['rho0']
    #
    #    for jj in range(0,len(kk_cosine)):
    #        for ii in range(0,len(kk)):
    #
    #            k = kk[ii]
    #            n = kk_cosine[jj]
    #            kvec = np.array([k,n])
    #            kmag = np.linalg.norm(kvec)
    #
    #            if kmag != 0:
    #                p['c'][ii,jj] = S['c'][ii,jj]*rho0*g*cs*n/kmag**2
    #
    #    #plt.contourf(p['g'])
    #    #plt.show()
    #    #pdb.set_trace()

    if ICsTestModulation == 1:
        S['g'] = np.sin(np.pi/Lz*z)


# Integration parameters
dir_state = 'State'
if Restart == 1:
    # Load restart file
    write, dt = solver.load_state('Results/' + dir_state + '/State_s10.h5', -1)
else:
    dt = 1./600.

SimDays = 0.
SimHrs = 0.
SimMins = 0.
SimSecs = 1.
te = SimDays*(24*60*60) + SimHrs*(60*60) + SimMins*60 + SimSecs
solver.stop_sim_time = te

Days = 5.
Hrs = 0.
Mins = 0.
Secs = 0.
wall_time = Days*(24*60*60) + Hrs*(60*60) + Mins*60 + Secs
solver.stop_wall_time = wall_time 
solver.stop_iteration = np.inf

#Set data write frequency.
if w2f_state == 1 or w2f_energy: write_dt = 1.0/10
if w2f_SinglePoint == 1: write_dt = 1.0/100

# Analysis:
file_nt = 60./write_dt	#Each file contains 1 min of data
Results = solver.evaluator.add_file_handler('Results/' + dir_state, sim_dt=write_dt, max_writes=file_nt, mode='append')

if w2f_state == 1:
    Results.add_system(solver.state)
    if CoordinateRotation == 1: 
        Results.add_task("GF_cr(S,psi,t,0)", layout='g', name='S_r')
        Results.add_task("GF_cr(S,psi,t,1)", layout='g', name='psi_r')

if w2f_energy == 1:
    #Potential Energy:
    #PE_L = "integ( g*cs*bs*w*z, 'x','z')"
    #Results.add_task(PE_L, layout='g', name='PE_L')
    #if Inviscid == 0: 
    #    PE_diff = "integ( g*cs*kappas*(d(S,x=2)+d(S,z=2))*z, 'x','z')"
    #    Results.add_task(PE_diff, layout='g', name='PE_diff')
    #if Linear == 0: 
    #    PE_adv = "integ( -(u*dx(g*cs*S*z) + w*dz(g*cs*S*z)), 'x','z')"
    #    Results.add_task(PE_adv, layout='g', name='PE_adv')
    #PE_N = "integ( g*cs*S*w, 'x','z')"
    #Results.add_task(PE_N, layout='g', name='PE_N')
    #Kinetic Energy:
    #KE_b = "-integ( g*cs*S*w, 'x','z')"
    #KE_p = "-integ( 1/rho0*(u*dx(p) + w*dz(p)), 'x','z')"
    #Results.add_task(KE_b, layout='g', name='KE_b')
    #Results.add_task(KE_p, layout='g', name='KE_p')
    #if Inviscid == 0:
    #    KE_diff = "integ( nu*( u*d(u,x=2) + u*d(u,z=2) + w*d(w,x=2) + w*d(w,z=2) ) , 'x','z')"
    #    Results.add_task(KE_diff, layout='g', name='KE_diff')
    #if Linear == 0:
    #    KE_adv = "-integ( u*dx((u**2+w**2)/2) + w*dz((u**2+w**2)/2), 'x','z')"
    #    Results.add_task(KE_adv, layout='g', name='KE_adv')
    #Analyse horizontal/vertical KE:
    #KE_x_xz = "-1/rho0*u*dx(p)"
    #KE_z_xz = "-1/rho0*w*dz(p) - cs*g*w*S"
    #if Inviscid == 0: 
    #    KE_x_xz += "+nu*u*L(u)" 
    #    KE_x_xz += "+nu*w*L(w)"
    #if Linear == 0: 
    #    KE_x_xz += "-u*(u*dx(u)+w*dz(u))"
    #    KE_z_xz += "-w*(u*dx(w)+w*dz(w))"
    #KE_x = "integ(" + KE_x_xz + ", 'x','z')" 
    #KE_z = "integ(" + KE_z_xz + ", 'x','z')"
    #Results.add_task(KE_x, layout='g', name='KE_x')
    #Results.add_task(KE_z, layout='g', name='KE_z')
    #Check pressure calculation at t=0:
    #p_check = "L(p)+g*cs*rho0*dz(S)"
    #Results.add_task(p_check, layout='g', name='p_check')

    #Potential Energy:
    PE_tot = "integ( z*cs*g*S, 'x','z')"
    PE_L = "integ( g*cs*bs*w*z, 'x','z')"
    #Kinetic Energy:
    KE_tot = "integ( 0.5*(u**2 + w**2), 'x','z')"
    Results.add_task(PE_tot, layout='g', name='PE_tot')
    Results.add_task(PE_L, layout='g', name='PE_L')
    Results.add_task(KE_tot, layout='g', name='KE_tot')


if w2f_SinglePoint == 1:
    xvec = np.loadtxt('XGridDedalus.txt')
    zvec = np.loadtxt('ZGridDedalus.txt')
    xpnt = xvec[int(Nx/2.)]
    zpnt = zvec[int(Nz/2.)]
    Results.add_task(de.operators.interpolate(solver.state['S'], x=xpnt, z=zpnt), name='S')
    Results.add_task(de.operators.interpolate(solver.state['T'], x=xpnt, z=zpnt), name='T')
    Results.add_task(de.operators.interpolate(solver.state['psi'], x=xpnt, z=zpnt), name='psi')
    if CoordinateRotation == 1:
        solver.evaluator.vars['xpnt'] = xvec[int(Nx/2.)]
        solver.evaluator.vars['zpnt'] = zvec[int(Nz/2.)]
        Results.add_task("interp(GF_cr(S,psi,t,0), x=xpnt,z=zpnt)", layout='g', name='S_r')
        Results.add_task("interp(GF_cr(S,psi,t,1), x=xpnt,z=zpnt)", layout='g', name='psi_r')

    #The following fails when running in parallel:
    #xIdx = int(Nx/2.)
    #zIdx = int(Nz/2.)
    #Results.add_task(solver.state['S']['g'][xIdx,zIdx], name='S')
    #Results.add_task(solver.state['S']['g'][xIdx,zIdx], name='psi')

if w2f_dt == 1:
    #Open file to write out clocktime, simulation time, and timestep:
    if Restart == 1: 
        fileDt = open("Results/" + dir_state + "/dt_" + str(solver.iteration) + ".txt","w")
        NextWriteT = solver.sim_time
    else: 
        fileDt = open("Results/" + dir_state + "/dt.txt","w")
        NextWriteT = 0.
    WriteDt = write_dt


# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, max_dt=write_dt, cadence=1, safety=1., threshold=0.1)
CFL.add_velocities(('u', 'w'))


# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:       

        if (w2f_dt == 1) and (solver.sim_time >= NextWriteT):
            fileDt.write(str(solver.sim_time) + ', ' + str(dt) + '\n')
            fileDt.flush()
            os.fsync(fileDt.fileno())
            NextWriteT += WriteDt

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

fileDt.close()
