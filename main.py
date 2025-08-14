import psutil
import time

def estimate_power(tdp=65, duration=10,interval=1, emission_factor=0.708):
    total_power = 0
    total_energy=0
    for _ in range(int(duration/interval)):
        cpu_usage = psutil.cpu_percent(interval=1)
        power = (cpu_usage / 100) * tdp
        
        energy=power*interval
        print(f"CPU Usage: {cpu_usage}% ,~{power:.2f} W ,~{energy:.2f} J")
        total_power += power
        total_energy+=energy
        total_energy_wh = total_energy / 3600
        total_energy_kwh = total_energy_wh / 1000
        total_emissions_kg = total_energy_kwh * emission_factor
    print(f"Average Estimated Power: {total_power/duration:.2f} W")
    print(f"Average Estimated Energy: {total_energy/duration:.2f} J or {total_energy_wh/duration:.2f} Wh")
    print(f"Estimated Carbon Emissions: {total_emissions_kg*1000:.3f} gCO₂e")

estimate_power(tdp=65, duration=10,interval=1,emission_factor=0.708)  