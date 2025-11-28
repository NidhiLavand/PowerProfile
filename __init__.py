import time
import threading
import statistics
import math
from contextlib import contextmanager

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

# NVML (NVIDIA Management Library) for GPU power readings
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class EnergyMeter:

    def __init__(self,
                 sample_interval=0.1,
                 cpu_tdp_w=65.0,
                 gpu_tdp_w=None,
                 carbon_intensity_g_per_kwh=400.0):
    
        self.sample_interval = float(sample_interval)
        self.cpu_tdp_w = float(cpu_tdp_w)
        self.gpu_tdp_w = float(gpu_tdp_w) if gpu_tdp_w is not None else None
        self.carbon_intensity = float(carbon_intensity_g_per_kwh)

        # detection
        self.device = self._detect_device()
        self.gpu_handle = None
        if self.device == 'gpu' and _NVML_AVAILABLE:
            self.gpu_handle = self._get_nvml_handle()

        # CPU RAPL handles (list of file paths for energy_uj)
        self._rapl_paths = self._discover_rapl_paths()

        # internal storage for metrics
        self._batch_records = []
        self._epoch_records = []

    # detection helpers
    def _detect_device(self):
        # GPU if PyTorch says cuda available or NVML found GPUs
        try:
            if torch is not None and torch.cuda.is_available():
                return 'gpu'
        except Exception:
            pass
        if _NVML_AVAILABLE:
            try:
                count = pynvml.nvmlDeviceGetCount()
                if count > 0:
                    return 'gpu'
            except Exception:
                pass
        return 'cpu'

    def _get_nvml_handle(self):
        try:
            # use device 0 by default
            return pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None

    def _get_gpu_name(self):
        if self.gpu_handle is not None:
            try:
                return pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
            except Exception:
                try:
                    return pynvml.nvmlDeviceGetName(self.gpu_handle)
                except Exception:
                    return 'nvidia-gpu'
        # fallback
        if torch is not None and torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return 'cuda-device'
        return None

    def _discover_rapl_paths(self):
        # Look for files like /sys/class/powercap/intel-rapl:0/energy_uj
        import glob
        paths = glob.glob('/sys/class/powercap/**/energy_uj', recursive=True)
        if paths:
            return paths
        return []

    # hardware reading 
    def _read_nvml_power_w(self):
        if not _NVML_AVAILABLE or self.gpu_handle is None:
            return None
        try:
            # nvmlDeviceGetPowerUsage returns milliwatts
            mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
            return mw / 1000.0
        except Exception:
            return None

    def _read_rapl_energy_uj(self):
        # returns total energy in microjoules across all discovered RAPL domains
        total = 0
        ok = False
        for p in self._rapl_paths:
            try:
                with open(p, 'r') as f:
                    v = int(f.read().strip())
                    total += v
                    ok = True
            except Exception:
                continue
        if ok:
            return total
        return None

    # sampling loop 
    def _sampler(self, stop_event, readings):
        # collects (timestamp, cpu_w, gpu_w) pairs until stop_event is set
        while not stop_event.is_set():
            t = time.time()
            cpu_w = self._estimate_cpu_power_w()
            gpu_w = self._read_nvml_power_w()
            readings.append((t, cpu_w, gpu_w))
            time.sleep(self.sample_interval)

    # estimation fallback 
    def _estimate_cpu_power_w(self):
        # preferred: use psutil (cpu_percent) to scale TDP
        try:
            if psutil is not None:
                util = psutil.cpu_percent(interval=None)
                # util is 0-100, assume linear scaling of TDP
                return (util / 100.0) * self.cpu_tdp_w
        except Exception:
            pass
        # last resort: assume idle-ish power
        return self.cpu_tdp_w * 0.2

    # high level APIs
    def measure(self, func, *args, **kwargs):
       
        with self._measure_block() as m:
            res = func(*args, **kwargs)
        m['result'] = res
        return res, m

    @contextmanager
    def _measure_block(self, label=None):
        # generic context manager used by batch() and epoch()
        readings = []
        stop_event = threading.Event()
        sampler_thread = threading.Thread(target=self._sampler, args=(stop_event, readings), daemon=True)

        # RAPL start
        rapl_start = self._read_rapl_energy_uj()

        t0 = time.time()
        sampler_thread.start()
        try:
            yield {}  # caller may perform work
        finally:
            # stop sampler and compute metrics
            stop_event.set()
            sampler_thread.join()
            t1 = time.time()
            rapl_end = self._read_rapl_energy_uj()

            duration = t1 - t0

            # compute integrated energy (Joules) by trapezoid on available readings
            # readings entries: (timestamp, cpu_w, gpu_w)
            energy_cpu_j = None
            energy_gpu_j = None
            total_energy_j = None

            if readings:
                # pre-process replace None with fallback estimates for gpu
                processed = []
                for (ts, cpu_w, gpu_w) in readings:
                    if cpu_w is None:
                        cpu_w = self._estimate_cpu_power_w()
                    if gpu_w is None:
                        # if we have a gpu_tdp fallback use that else 0
                        gpu_w = float(self.gpu_tdp_w) if self.gpu_tdp_w is not None else 0.0
                    processed.append((ts, float(cpu_w), float(gpu_w)))

                # integrate cpu and gpu separately
                cpu_energy = 0.0
                gpu_energy = 0.0
                for i in range(1, len(processed)):
                    t_prev, cpu_prev, gpu_prev = processed[i - 1]
                    t_cur, cpu_cur, gpu_cur = processed[i]
                    dt = t_cur - t_prev
                    # trapezoid
                    cpu_energy += (cpu_prev + cpu_cur) / 2.0 * dt
                    gpu_energy += (gpu_prev + gpu_cur) / 2.0 * dt

                energy_cpu_j = cpu_energy
                energy_gpu_j = gpu_energy
                total_energy_j = cpu_energy + gpu_energy

            # If RAPL counters exist use them (more accurate for CPU); RAPL is given in micro-joules
            if rapl_start is not None and rapl_end is not None:
                rapl_delta_uj = rapl_end - rapl_start
                # some counters wrap — try to handle by assuming monotonic small wrap
                if rapl_delta_uj < 0:
                    # assume 32 or 64-bit wrap; add 2**32*1e6 or 2**64*1e6 — naive but best-effort
                    rapl_delta_uj += 2 ** 32
                cpu_j_from_rapl = rapl_delta_uj / 1e6
                # if we already computed cpu energy from sampling, trust rapl more
                energy_cpu_j = cpu_j_from_rapl
                if total_energy_j is None:
                    total_energy_j = energy_cpu_j
                else:
                    # replace cpu portion with rapl value
                    # keep gpu portion from sampler
                    if energy_gpu_j is not None:
                        total_energy_j = energy_gpu_j + energy_cpu_j

            # If no sampler readings present, fall back to TDP*duration
            if total_energy_j is None:
                # fallback
                est_cpu = self.cpu_tdp_w * duration
                est_gpu = (self.gpu_tdp_w if self.gpu_tdp_w is not None else 0.0) * duration
                energy_cpu_j = est_cpu
                energy_gpu_j = est_gpu
                total_energy_j = est_cpu + est_gpu

            avg_power = total_energy_j / duration if duration > 0 else 0.0
            carbon_g = (total_energy_j / 3600.0) * self.carbon_intensity  # J -> Wh -> kWh then grams

            metrics = dict(
                label=label,
                duration_s=duration,
                energy_j=total_energy_j,
                energy_cpu_j=energy_cpu_j,
                energy_gpu_j=energy_gpu_j,
                avg_power_w=avg_power,
                carbon_g=carbon_g,
            )

            # append to proper store based on label
            if label == 'batch' or label is None:
                self._batch_records.append(metrics)
            elif label == 'epoch':
                self._epoch_records.append(metrics)

            # yield metrics to caller if they want to inspect after the with-block
            # (we returned an empty dict earlier; but user can call meter.last_batch() etc.)
            # For convenience also return metrics object via attribute
            self._last_metrics = metrics

    @contextmanager
    def batch(self):
  
        cm = self._measure_block(label='batch')
        with cm as ctx:
            yield ctx

    @contextmanager
    def epoch(self):
        cm = self._measure_block(label='epoch')
        with cm as ctx:
            yield ctx

    # -------------------- reporting --------------------
    def last_batch(self):
        return getattr(self, '_last_metrics', None)

    def batch_summary(self):
        return self._aggregate(self._batch_records)

    def epoch_summary(self):
        return self._aggregate(self._epoch_records)

    def summary(self):
        return dict(
            device=self.device,
            gpu_name=self._get_gpu_name(),
            batch_summary=self.batch_summary(),
            epoch_summary=self.epoch_summary(),
            num_batches=len(self._batch_records),
            num_epochs=len(self._epoch_records),
            carbon_intensity_g_per_kwh=self.carbon_intensity,
        )

    def _aggregate(self, records):
        if not records:
            return None
        durations = [r['duration_s'] for r in records]
        energies = [r['energy_j'] for r in records]
        carbons = [r['carbon_g'] for r in records]
        avg = dict(
            count=len(records),
            total_duration_s=sum(durations),
            mean_duration_s=statistics.mean(durations),
            total_energy_j=sum(energies),
            mean_energy_j=statistics.mean(energies),
            total_carbon_g=sum(carbons),
            mean_carbon_g=statistics.mean(carbons),
        )
        return avg


#  small self-test / example 
if __name__ == '__main__':
    # Quick demo of usage. Replace the `work()` function with your training step.
    def work(duration=1.0):
        # emulate CPU + GPU work by burning time
        t_end = time.time() + duration
        x = 0
        while time.time() < t_end:
            x += 1
        return x

    meter = EnergyMeter(sample_interval=0.05, cpu_tdp_w=65.0, gpu_tdp_w=200.0, carbon_intensity_g_per_kwh=450.0)
    print('Detected device:', meter.device)
    if meter.device == 'gpu':
        print('GPU name:', meter._get_gpu_name())

    for epoch in range(2):
        with meter.epoch():
            for b in range(3):
                with meter.batch():
                    work(0.3)
                print('Batch metrics:', meter.last_batch())
        print('Epoch summary so far:', meter.epoch_summary())

    print('\nOverall summary:')
    import pprint
    pprint.pprint(meter.summary())


