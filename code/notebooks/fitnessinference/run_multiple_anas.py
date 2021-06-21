import numpy as np
import csv
import os
import copy
import pickle
import logging
from datetime import date
from general.queuing import QsubHeader, SlurmHeader, run_sbatch
import time

# Writes Slurm files to be run on the cluster
class SlurmProtocol(object):
    def __init__(self, simulation_time=2000, nodes=1, ppn=1):
        self.header = SlurmHeader(simulation_name="fluSimulation", simulation_time=simulation_time,
                                  nodes=nodes, ppn=ppn)

    def set_python_script(self, q):
        pypath = os.path.normpath('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/NewApproachFromMarch2021/InfluenzaFitnessInference/code/notebooks/fitnessinference/simulation.py')
        if not os.path.exists(pypath):
            pypath = os.path.join(os.getcwd(), 'code', 'notebooks', 'fitnessinference', 'analysis.py')
        command_string = 'python ' + pypath + '\n'

        q.write(command_string)

    def generate_slurm(self):
        q = open("sbatch.sh", "w")
        self.header.set_header(q)
        self.set_python_script(q)
        q.close()

def main():
    # run analyses on cluster
    slurm = SlurmProtocol()
    slurm.generate_slurm()
    run_sbatch()
    # time.sleep(1) # wait for 100 seconds so that result file gets created before next simu is run

# if this file is run from the console, the function main will be executed
if __name__ == '__main__':
    main()
