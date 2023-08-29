import os
import sys
import time

from python.disk_experiment import run_experiment
from python.telemetry import collect_telemetry


def main(argv):
    print("Starting telemetry thread...")

    ppid = os.getpid()
    pid = os.fork()
    if pid == 0:
        collect_telemetry(ppid, "metrics/{}_telemetry.txt".format(argv[1]), 1)
        exit(0)

    time.sleep(6)
    print("Running experiment...")
    run_experiment(int(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
