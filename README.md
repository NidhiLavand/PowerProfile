

# Energy-Meter: Lightweight Energy and Carbon Instrumentation for ML

## Project Overview

`energy-meter` is a minimal, yet powerful Python utility designed to instrument Machine Learning training loops and measure the **power consumption, energy usage, and estimated carbon emissions** of your code.

It automatically detects and prioritizes hardware-level counters (like NVIDIA NVML and Intel RAPL) for maximum accuracy, gracefully falling back to efficient TDP-based estimations when hardware access is unavailable.

## Features

  * **Hardware Prioritization:** Directly reads instantaneous power from **NVIDIA GPUs** via `pynvml` and **CPU energy** from Intel RAPL (`/sys/class/powercap`).
  * **Intelligent Fallback:** When hardware counters are unavailable, it uses CPU utilization (`psutil.cpu_percent`) and user-defined/default **TDP (Thermal Design Power)** values to provide conservative power estimations.
  * **Context Managers:** Offers simple `with meter.batch():` and `with meter.epoch():` blocks for easy, granular measurement during training.
  * **Carbon Footprint Estimation:** Calculates **grams of $\text{CO}_2$** emitted based on the total energy consumed and a configurable **carbon intensity** (g CO2/kWh).
  * **Comprehensive Reporting:** Provides detailed metrics including duration, total energy (J), average power (W), and carbon (g) for batches and epochs.

## Installation

You can install the package via `pip`.

```bash
pip install energy-meter
```

### Recommended Dependencies

For **hardware-level power readings** (maximum accuracy), it is recommended to install the following optional packages:

```bash
# For NVIDIA GPU power
pip install pynvml
# For CPU utilization and general system info
pip install psutil
# If using PyTorch (helps with device detection)
pip install torch
```

## Usage Example

The primary class is `EnergyMeter`, which you initialize once and use its context managers (`.batch()` and `.epoch()`) to scope your measurements.

```python
from energy_meter import EnergyMeter
import time

# 1. Initialize the meter
# You can customize the carbon intensity (e.g., 400 g CO2/kWh for your region)
meter = EnergyMeter(
    sample_interval=0.05, cpu_tdp_w=65.0, gpu_tdp_w=200.0, carbon_intensity_g_per_kwh=450.0  # sample power every 0.1 seconds
)

print(f"Device Detected: {meter.device.upper()}")
if meter.device == 'gpu':
    print(f"GPU Name: {meter._get_gpu_name()}")

# Dummy function to simulate a training step
def train_step(duration):
    """Simulates work being done by burning CPU time."""
    t_end = time.time() + duration
    while time.time() < t_end:
        pass # Placeholder for actual ML code (e.g., model(batch).backward().step())


# 2. Wrap your training loop with context managers
for epoch in range(2):
    with meter.epoch(): # Measures the entire epoch
        for batch_idx in range(3):
            with meter.batch(): # Measures a single batch step
                train_step(0.3) # Simulate a 0.3 second batch
            
            # Optional: Inspect results after each batch
            last_metrics = meter.last_batch()
            print(f"Epoch {epoch} Batch {batch_idx}: {last_metrics['energy_j']:.2f} J, {last_metrics['carbon_g']:.4f} gCO2")

# 3. Print the final summary
print("\n--- Final Summary ---")
import pprint
pprint.pprint(meter.summary())
```
The red highlighting you see is likely due to how you've tried to combine the JSON output and the surrounding Markdown/text, causing a formatting error that your local editor's linter or spell checker is flagging. Also, the LaTeX equation output seems to have failed to render correctly outside of the text block in your environment.

Here is the corrected and cleaned-up README section. I've separated the JSON output from the surrounding text and ensured the equation is correctly formatted for GitHub Markdown.

-----

## 📊 Summary Output Structure

The `meter.summary()` method returns a dictionary containing high-level metrics aggregated over all measured batches and epochs:

```json
{
    'batch_summary': {'count': 6,
                   'mean_carbon_g': 7.797722575701774,
                   'mean_duration_s': 0.31085503101348877,
                   'mean_energy_j': 62.38178060561419,
                   'total_carbon_g': 46.78633545421064,
                   'total_duration_s': 1.8651301860809326,
                   'total_energy_j': 374.2906836336851},
 'carbon_intensity_g_per_kwh': 450.0,
 'device': 'cpu',
 'epoch_summary': {'count': 2,
                   'mean_carbon_g': 28.2488715724647,
                   'mean_duration_s': 0.967894434928894,
                   'mean_energy_j': 225.9909725797176,
                   'total_carbon_g': 56.4977431449294,
                   'total_duration_s': 1.935788869857788,
                   'total_energy_j': 451.9819451594352},
 'gpu_name': None,
 'num_batches': 6,
 'num_epochs': 2
}
```

-----

## ⚙️ How Measurement Works

1.  **Start:** When entering the `with meter.batch()` block, a background **sampler thread** starts.
2.  **Sampling:** The thread runs every `sample_interval` (default 0.1s), recording the instantaneous power (in Watts) for the CPU and GPU.
      * **GPU Power:** Read via `pynvml.nvmlDeviceGetPowerUsage()`.
      * **CPU Energy (Preferred):** The initial **Intel RAPL** energy counter value ($\mu$J) is read from `/sys/class/powercap`.
3.  **Stop & Integrate:** When exiting the block, the final RAPL value is read, and the total CPU energy (J) is calculated from the difference.
4.  **Integration:** For components relying on power sampling (like the GPU), energy (J) is calculated by integrating the sampled power over time using the **trapezoidal rule** ($\sum \text{Power}_{avg} \cdot \Delta t$).
5.  **Carbon Calculation:** Energy in Joules (J) is converted to Kilowatt-hours (kWh), and then multiplied by the provided carbon intensity ($\text{g}/\text{kWh}$) to get grams of $\text{CO}_2$ (g).

$$\text{Energy}_{\text{J}} \xrightarrow{\text{convert}} \text{Energy}_{\text{kWh}} \times \text{Carbon Intensity}_{\text{g/kWh}} = \text{Carbon}_{\text{g}}$$
