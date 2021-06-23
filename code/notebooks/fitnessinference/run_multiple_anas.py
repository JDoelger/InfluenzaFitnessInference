import numpy as np
import copy
import os
from pypet import Trajectory, cartesian_product
import pickle
import scipy
try:
    import simulation as simu
except ModuleNotFoundError:
    from fitnessinference import simulation as simu
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
import matplotlib as mpl
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
import logging
from datetime import date
from general.queuing import QsubHeader, SlurmHeader, run_sbatch
import time

# Writes Slurm files to be run on the cluster
class SlurmProtocol(object):
    def __init__(self, simulation_time=2000, nodes=1, ppn=4, mem_gb=10):
        self.header = SlurmHeader(simulation_name="fluSimulation", simulation_time=simulation_time,
                                  nodes=nodes, ppn=ppn, mem_gb=mem_gb)

    def set_python_script(self, q):
        pypath = os.path.normpath('C:/Users/julia/Documents/Resources/InfluenzaFitnessLandscape/NewApproachFromMarch2021/'
                                  'InfluenzaFitnessInference/code/notebooks/fitnessinference/analysis.py')
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
    # time.sleep(1) # wait for x seconds so that result file gets created before next simu is run

# if this file is run from the console, the function main will be executed
if __name__ == '__main__':
    main()
