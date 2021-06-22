#!/bin/bash
#SBATCH --job-name=fluSimulation
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time=1-9:20:00
#SBATCH --partition=sched_mit_arupc_long

#SBATCH --mem-per-cpu=10gb

python /home/jdoelger/InfluenzaFitnessInference/code/notebooks/fitnessinference/analysis.py
