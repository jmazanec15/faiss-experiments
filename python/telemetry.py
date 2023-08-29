import time

import psutil


def collect_telemetry(pid, file_output, frequency):
    p = psutil.Process(pid)
    with open(file_output, 'w') as f:
        while True:
            time.sleep(frequency)
            if not psutil.pid_exists(pid):
                exit(0)
            f.write("{},{}\n".format(p.io_counters().read_count, p.memory_full_info().rss))
            f.flush()