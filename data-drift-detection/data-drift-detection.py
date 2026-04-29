import numpy as np

def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    reference_counts = np.array(reference_counts)
    production_counts = np.array(production_counts)
    ref_sum = sum(reference_counts)
    prod_sum = sum(production_counts)
    reference_counts = reference_counts / ref_sum
    production_counts = production_counts / prod_sum
    tvd = 0.5 * sum(np.abs(production_counts - reference_counts))
    return {"score": tvd, "drift_detected": bool(tvd > threshold)}