"""
All metric computations: PDR, delay CDF, energy, convergence.
Ref: Mega-Prompt Evaluation Metrics
"""
def compute_pdr(sent, delivered):
    return delivered / max(1, sent)

def compute_average_delay(delays):
    return sum(delays) / max(1, len(delays))

def compute_energy_efficiency(total_throughput, total_power_watts):
    return total_throughput / max(1e-9, total_power_watts)
