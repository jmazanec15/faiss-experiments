import os
import sys
import time

from python.experiments import disk_experiment, recall_experiment
from python.utils.telemetry import collect_telemetry


def main(argv):
    print("Starting telemetry thread...")

    ppid = os.getpid()
    pid = os.fork()
    if pid == 0:
        collect_telemetry(ppid, "metrics/{}_telemetry.txt".format(argv[1]), 5)
        exit(0)

    time.sleep(6)
    print("Running experiment...")
    # disk_experiment.run_experiment(int(argv[1]))
    recall_experiment.run_experiment(int(argv[1]))

if __name__ == "__main__":
    main(sys.argv)
