#!/bin/sh
#PBS -d .
#PBS -q pq 
#PBS -l walltime=00:60:00 
#PBS -A Research_Project-183035 
#PBS -l nodes=1:ppn=8 
#PBS -m e -M p.burns2@exeter.ac.uk

echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'
echo PBS_O_HOST      = $PBS_O_HOST
echo PBS_ENVIRONMENT = $PBS_ENVIRONMENT
echo PBS_NODEFILE    = $PBS_NODEFILE
echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'

FiredrakePath="/gpfs/ts0/home/pb412/firedrake"
GustoPath="$FiredrakePath/src/gusto"
ExecPath="$GustoPath/examples/boussinesq_2d_lab.py"

module purge
module load Python/3.5.2-foss-2016b
. "$FiredrakePath/bin/activate"
export OMPI_MCA_mpi_warn_on_fork=0

echo -e 'Running Firedrake...'
mpirun -np 8 python3 $ExecPath

#mpirun \
#-x FIREDRAKE_TSFC_KERNEL_CACHE_DIR \
#-x PYOP2_CACHE_DIR \
#-x LIBRARY_PATH \
#-x LD_LIBRARY_PATH \
#-x CPATH \
#-x VIRTUAL_ENV \
#-x PATH \
#-x PKG_CONFIG_PATH \
#-x PYTHONHOME \
#-x PYTHONPATH \
#-np 16 python $EXEC > log


##PBS -V # export all environment variables to the batch job.
