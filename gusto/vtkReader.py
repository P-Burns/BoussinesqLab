#September 2018
#Program to read Gusto-output VTK files into python numpy arrays for advanced data analysis.


#Load required libraries
from paraview.simple import *
from vtkmodules.util import numpy_support
import numpy as np
import pdb as pdb


#Choose input file:
fnm = './results/tmp_login/field_output_0.pvtu'


#Ask ParaView to read data in input file by 
#assigning the correct reader for the given file format.
#The return value is a so-called ParaView server object.
serverObj = paraview.simple.OpenDataFile(fnm)

#Pass data from server to client:
clientObj = servermanager.Fetch(serverObj)

#Extract buoyancy data only from client object:
data = clientObj.GetPointData().GetArray('b')

Np = int(clientObj.GetNumberOfPoints())

#Convert to numpy array for analysis:
b = numpy_support.vtk_to_numpy(data)
type(b)
np.shape(b)


pdb.set_trace()
