#!/bin/sh
#PBS -d .
#PBS -q ptq 
#PBS -l walltime=00:10:00 
#PBS -A Research_Project-183035 
#PBS -l nodes=1:ppn=4
#PBS -m e -M p.burns2@exeter.ac.uk

echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'
echo PBS_O_HOST      = $PBS_O_HOST
echo PBS_ENVIRONMENT = $PBS_ENVIRONMENT
echo PBS_NODEFILE    = $PBS_NODEFILE
echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'

FiredrakePath="/gpfs/ts0/home/pb412/firedrake"
GustoPath="$FiredrakePath/src/gusto"
ExecPath="$GustoPath/examples/boussinesq_3d_lab.py"

module purge
module load Python/3.5.2-foss-2016b
. "$FiredrakePath/bin/activate"
export OMPI_MCA_mpi_warn_on_fork=0

echo -e 'Running Firedrake...'
time mpirun --tag-output -np 4 python3 $ExecPath > log.out 2>&1
