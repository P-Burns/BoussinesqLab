#!/bin/bash
#SBATCH --export=ALL # export all environment variables to the batch job
#SBATCH -D . # set working directory to .
#SBATCH -p ptq
#SBATCH --time=00:10:00 # maximum walltime for the job
#SBATCH -A Research_Project-183035 # research project to submit under
#SBATCH --nodes=1 # specify number of nodes
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=p.burns2@exeter.ac.uk

echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'
echo PBS_O_HOST      = $SLURM_SUBMIT_HOST
echo PBS_NODEFILE    = $SLURM_JOB_NODELIST
echo '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'

DedalusPath="/gpfs/ts0/home/pb412/dedalus"
ExecPath="/gpfs/ts0/home/pb412/BoussinesqLab/dedalus/plotMultipleN.bsh"

module purge
module load Python/3.5.2-foss-2016b
. "$DedalusPath/bin/activate"
export OMPI_MCA_mpi_warn_on_fork=0

echo -e 'Running Dedalus...'
mpirun -np 16 python3 $ExecPath
