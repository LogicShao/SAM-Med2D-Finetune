"""Low-overhead runtime sampling for single-GPU training experiments."""

import statistics
import subprocess
import threading
import time


class GpuUtilizationMonitor:
    def __init__(self, gpu_index, interval_seconds=1.0):
        self.gpu_index = int(gpu_index)
        self.interval_seconds = max(float(interval_seconds), 0.5)
        self.samples = []
        self.error = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return self.summary()
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 2.0)
        return self.summary()

    def _run(self):
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_seconds)

    def _sample_once(self):
        command = [
            "nvidia-smi",
            "--id",
            str(self.gpu_index),
            "--query-gpu=utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            values = [value.strip() for value in completed.stdout.strip().split(",")]
            if len(values) != 3:
                raise ValueError("Unexpected nvidia-smi output: {}".format(completed.stdout.strip()))
            self.samples.append(
                {
                    "timestamp": time.time(),
                    "gpu_utilization_percent": float(values[0]),
                    "memory_utilization_percent": float(values[1]),
                    "memory_used_mib": float(values[2]),
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.error = str(exc)

    def summary(self):
        if not self.samples:
            return {
                "enabled": True,
                "gpu_index": self.gpu_index,
                "sample_count": 0,
                "error": self.error,
            }

        summary = {
            "enabled": True,
            "gpu_index": self.gpu_index,
            "sample_count": len(self.samples),
            "interval_seconds": self.interval_seconds,
            "error": self.error,
        }
        for key in ("gpu_utilization_percent", "memory_utilization_percent", "memory_used_mib"):
            values = [sample[key] for sample in self.samples]
            summary["{}_mean".format(key)] = statistics.mean(values)
            summary["{}_max".format(key)] = max(values)
        return summary


def parse_cuda_device_index(device):
    value = str(device)
    if not value.startswith("cuda"):
        return None
    if ":" not in value:
        return 0
    return int(value.split(":", 1)[1])
