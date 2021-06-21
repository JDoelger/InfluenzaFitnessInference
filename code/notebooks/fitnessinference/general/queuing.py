import datetime
import os
import subprocess


def run_qsub():
    (stdout, stderr) = subprocess.Popen(["qsub {0}".format("qsub.sh")], shell=True, stdout=subprocess.PIPE,
                                        cwd=os.getcwd()).communicate()
    f = open("error_log", "w")
    f.write("stdout = " + str(stdout) + "\n")
    f.write("stderr = " + str(stderr) + "\n")
    f.close()


class QsubHeader(object):
    def __init__(self, simulation_name, simulation_time=30, nodes=1, ppn=1):
        self.simulation_name = simulation_name
        self.simulation_time = simulation_time
        self.nodes = nodes
        self.ppn = ppn

    def set_qsub_header(self, q):
        q.write("#PBS -m ae\n")
        q.write("#PBS -q short\n")
        # q.write("#PBS -q forever\n")
        # q.write("#PBS -p 999\n")
        q.write("#PBS -V\n")
        q.write("#PBS -l walltime={0},nodes={1}:ppn={2} -N {3}\n\n".format(datetime.timedelta(minutes=self.simulation_time),
                                                                           self.nodes, self.ppn, self.simulation_name))

        q.write("cd $PBS_O_WORKDIR\n\n")
        q.write("echo $PBS_JOBID > job_id\n\n")


def run_sbatch():
    (stdout, stderr) = subprocess.Popen(["sbatch {0}".format("sbatch.sh")], shell=True, stdout=subprocess.PIPE,
                                        cwd=os.getcwd()).communicate()
    f = open("error_log", "w")
    f.write("stdout = " + str(stdout) + "\n")
    f.write("stderr = " + str(stderr) + "\n")
    f.close()
    #
    # return stdout, stderr


class SlurmHeader(object):
    def __init__(self, simulation_name, simulation_time=1500, nodes=1, ppn=1):
        self.simulation_name = simulation_name
        self.simulation_time = simulation_time
        self.nodes = nodes
        self.ppn = ppn

    def set_header(self, q):
        q.write("#!/bin/bash\n")
        q.write("#SBATCH --job-name={0}\n".format(self.simulation_name))
        q.write("#SBATCH --nodes {0}\n".format(self.nodes))
        q.write("#SBATCH --ntasks-per-node {0}\n".format(self.ppn))
        # q.write("#SBATCH --time={0}\n".format(datetime.timedelta(minutes=self.simulation_time)))
        q.write("#SBATCH --time=1-09:20:00") # set time in format as recommended by engaging admin
        q.write("#SBATCH --partition=sched_mit_arupc_long\n\n")
        # q.write("#SBATCH --mem-per-cpu=10gb\n\n")
