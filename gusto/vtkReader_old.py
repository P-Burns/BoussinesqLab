#Program to read Gusto-output VTK files into python numpy arrays.

#Field data in the VTK files is stored at grid vertices and data is repeated at repeated vertices. 
#I try two methods:
#1) remove duplication using duplicate coordinates, resort and reshape into numpy arrays
#2) interpolate from the vertices onto a unique set of points for b.


from paraview.simple import *
from vtkmodules.util import numpy_support
from paraview import python_view

import numpy as np
from scipy.interpolate import griddata
from itertools import product
import matplotlib.pyplot as plt
import pdb as pdb
from numpy import inf
import os


#Control section:
ReSort = 1
Interpolate = 0
NewGrid = 0
Method = "linear"
#Method = "cubic"
#Method = "nearest"
Check = 1
MeshTest = 0


#Loop over different simulation results (and times if required):
ts = 0
te = 0
nt = te-ts+1
dir_base = "./results/"
dir_vec = ["tmp_ubuntu5_pb","tmp_login"]
nruns = len(dir_vec)

#Define arrays to store Gusto output for later analysis:
Nx = 80
Nz = 180
results0 = np.zeros((Nx+1,Nz+1,nt))
if nruns == 2: results1 = np.zeros((Nx*2+1,Nz*2+1,nt))

for n in range(0,nruns):
    for t in range(0, nt):

        fnm = dir_base + dir_vec[n] + "/" + "field_output_" + str(t+ts) + ".pvtu"
        
	#Ask ParaView to read data in input file by 
	#assigning the correct reader for the given file format.
	#The return value is a so-called ParaView server object.
	reader = paraview.simple.OpenDataFile(fnm)
        #reader.ListProperties()
        reader.PointArrayStatus='b'
        #check:
        reader.GetPropertyValue('PointArrayStatus')

        #view = GetActiveViewOrCreate('RenderView')

        #dp = GetDisplayProperties() 
        #dp.Representation = 'Surface'
        #dp.RescaleTransferFunctionToDataRange(True)
        #readerRep = GetRepresentation()
        #readerRep.ColorArrayName = 'b' 
        #reader.UpdatePipeline()
        #Show(reader)
        #Render()
        
        #dir(serverObj)
        #serverObj.ListProperties()
        pdb.set_trace()
        view.SetAttributeArrayStatus(0, vtkDataObject.POINT, 'b', 1)
        pdb.set_trace()
 
	#Pass data from server to client:
	clientObj = servermanager.Fetch(serverObj)

        Np = int(clientObj.GetNumberOfPoints())	#this is the number of vertices in the grid
        #Nc = int(data.GetNumberOfCells()) 	#n.b. field values not available on central cell points

        #Get coordinates for each vertice:
        coord_arr = np.zeros((Np,2))
        for i in range(0,Np):
            coord_arr[i,:] = clientObj.GetPoint(i)[0:2]
            #VTK object assumes 3D spatial domain, so one coordinate (for y axis) is all zeros.
            #The y-coordinate is dropped here.
            #Note that vertice coordinates are repeated.

        #Vectors of unique coordinates for grid vertices:
        x1_unique = np.unique(coord_arr[:,0])
        z1_unique = np.unique(coord_arr[:,1])
        if n == 0: zvec0 = z1_unique
        if n == 1: zvec1 = z1_unique

        #Extract buoyancy data only from client object.
        #This is field data at cell vertices:
        b = clientObj.GetPointData().GetArray('b')

        b_arr = np.zeros(Np)
        for i in range(0,Np):
            b_arr[i] = b.GetTuple1(i)
            #This includes repeated values for repeated coordinates.

        #Remove duplication, sort and reshape data:
        if ReSort == 1:

            #To pre-sort data according to x-coordinate:            
            #sortIdxsX = np.argsort(coord_arr[:,0])
            #coord_arr = coord_arr[sortIdxsX,:]
            #b_arr = b_arr[sortIdxsX]

            for i in range(0,len(x1_unique)):

                #Find subset of points for chosen x-coordinate:
                idx_subset = np.where( coord_arr[:,0] == x1_unique[i] )
                idx_subset = np.asarray(idx_subset).flatten()
                subset = coord_arr[idx_subset,1].flatten()

                #Find unique set of z-points for x-coordinate subset:
                subsetUnique,idxUnique = np.unique(subset,return_index=True)

                #Sort unique points:
                sortIdxsZ = np.argsort(subsetUnique)
                tmp = subsetUnique[sortIdxsZ]

                #Perform same operations on b as applied to coordinates:
                subset_b = b_arr[idx_subset]
                subset_b_unique = subset_b[idxUnique]
                if n == 0: results0[i,:,t] = subset_b_unique[sortIdxsZ]
                if n == 1: results1[i,:,t] = subset_b_unique[sortIdxsZ]


            #Following is not used:
            #Find norms of all coordinate pairs:
            #norms = np.linalg.norm(coord_arr,axis=1)
            # 
            #sortIdxs = np.argsort(norms)
            #norms = np.sort(norms)
            #coord_arr = coord_arr[sortIdxs,:]
            #
            #Find tangent angles of all coordinate pairs:
            #angles = coord_arr[:,1]/coord_arr[:,0]
            #angles[np.isnan(angles)] = 999
            #angles[angles == inf] = 99999
            #
            #Find unique coordinates using norm and angle:
            #ids = norms+angles
            #ids_unique,idxs_unique = np.unique(ids,return_index=True)
            #coord_arr = coord_arr[idxs_unique,:]
            #
            #Subset unique values:
            #b_arr = b_arr[sortIdxs]
            #b_arr = b_arr[idxs_unique]

        if Interpolate == 1:

            if NewGrid == 0:
                Xgrid2, Zgrid2 = np.meshgrid(x1_unique,z1_unique, indexing='ij')
                #The coordinates need to be in this shape for the interpolation routine.
    
            if NewGrid == 1:
                #Read Gusto grid coordinates to enable interpolation from VTK object:
                GustoGrid = np.loadtxt('./GridGusto.txt')
                x2 = np.unique(GustoGrid[:,0])
                z2 = np.unique(GustoGrid[:,1])
                Xgrid2, Zgrid2 = np.meshgrid(x2,z2, indexing='ij')
                #The coordinates need to be in this shape for the interpolation routine.

            #Interpolate:
            results = griddata(coord_arr, b_arr, (Xgrid2, Zgrid2), method=Method)
            #Since interpolation coordinates exist in the list of datum coordinates, this 
            #effectively subsets a unique set of field values, as desired.

            #Store processed data for later analysis:
            if n == 0: results0[:,:,t] = results
            if n == 1: results1[:,:,t] = results


#Some code to check the interpolation for known t=0 buoyancy field:
if Check == 1:

    #The following is not used:
    #ParkRun = 18
    #if ParkRun == 14: N2=0.35
    #if ParkRun == 16: N2=1.34
    #if ParkRun == 18: N2=3.83
    #if ParkRun == 14: drho0_dz = -31.976
    #if ParkRun == 18: drho0_dz = -425.9
    #g = 9.81
    #rho0 = -g/N2*drho0_dz
    #drho0_dz13 = -122.09    
    #dgamma = 100./3
    #dz_b = 2./100
    #a0 = 100.
    #H = 0.45
    #z_a = H/2
    #rhoprime13 = dgamma*z_a + a0*dz_b + dgamma/2*dz_b
    #scalefactor = g/rho0* drho0_dz/drho0_dz13
    #bprime = rhoprime13 * scalefactor
    #RandomSample = np.loadtxt('./RandomSample_080_180.txt')
    #RandomSample = RandomSample/np.max(RandomSample)*bprime*5
    #diff = b_interp - RandomSample

    width = 10
    height = 20
    figure = python_view.matplotlib_figure(width, height)
    plt = figure.add_subplot(1,1,1)

    cmap='bwr'
    levels = 100
    plt.contourf(results0[:,:,0].transpose(), levels)
    #plt.colorbar()
    #if nruns == 2:
    #    plt.subplot(122)
    #    plt.contourf(results1[:,:,0].transpose(), levels, cmap=cmap)
    #    plt.colorbar()

    #plt.show()
    python_view.figure_to_image(figure)

if MeshTest == 1:

    plt.plot(results0[int(Nx/2.),:,0],zvec0,'b.-')
    plt.plot(results1[int(Nx*2/2.),:,0],zvec1, 'r.-')
    plt.show()
    




pdb.set_trace()
