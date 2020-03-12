#!/bin/sh
#PBS -d .
#PBS -q ptq 
#PBS -l walltime=00:10:00 
#PBS -A Research_Project-183035 
#PBS -l nodes=1:ppn=8 
#PBS -m e -M <add your email address>

echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'
echo PBS_O_HOST      = $PBS_O_HOST
echo PBS_ENVIRONMENT = $PBS_ENVIRONMENT
echo PBS_NODEFILE    = $PBS_NODEFILE
echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'

DedalusPath="/gpfs/ts0/home/<add your username>/dedalus"
ExecPath="/gpfs/ts0/home/<add your username>/<add path to executable (e.g. to elliptic.py)>"

module purge
module load Python/3.5.2-foss-2016b
. "$DedalusPath/bin/activate"
export OMPI_MCA_mpi_warn_on_fork=0

echo -e 'Running Dedalus...'
mpirun -np <add numbver of cores> python3 $ExecPath
