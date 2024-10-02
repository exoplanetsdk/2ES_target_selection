import math
import csv
from typing import Dict, List

def get_manual_values() -> Dict[str, float]:
    return {
        "Temp": float(input("Enter Temperature (K): ")),
        "Vmag": float(input("Enter V magnitude: ")),
        "MiDia": float(input("Enter Mirror Diameter (m): ")),
        "SpRes": float(input("Enter Spectral Resolution: ")),
        "TObs": float(input("Enter Observation Time (min): ")),
        "Lam_min": float(input("Enter minimum wavelength (nm): ")),
        "Lam_max": float(input("Enter maximum wavelength (nm): "))
    }

def get_wavelengths(all_rows: List[Dict]) -> Dict[str, List[float]]:
    lam_min = []
    lam_max = []
    lams = []
    lams_err = []

    for row in all_rows:
        lam_min.append(float(row['Lam_min']))
        lam_max.append(float(row['Lam_max']))
        lams.append((lam_min[-1] + lam_max[-1]) / 2.0)
        lams_err.append(lams[-1] - lam_min[-1])

    return {"Lambda": lams, "Lambda_err": lams_err}

def get_temperatures(header: List[str]) -> List[float]:
    return [float(temp) for temp in header if temp not in ['Lam_min', 'Lam_max']]

def get_rv_precision(all_rows: List[Dict], temperatures: List[float]) -> Dict[float, List[float]]:
    rv_precision = {temp: [] for temp in temperatures}
    for row in all_rows:
        for temp in temperatures:
            rv_precision[temp].append(float(row[str(temp)]))
    return rv_precision

def interpolate_rv_precision(rv_precision: Dict[float, List[float]], temperatures: List[float], target_temp: float) -> List[float]:
    if target_temp <= min(temperatures):
        return rv_precision[min(temperatures)]
    if target_temp >= max(temperatures):
        return rv_precision[max(temperatures)]
    
    for i in range(len(temperatures) - 1):
        if temperatures[i] <= target_temp < temperatures[i+1]:
            t1, t2 = temperatures[i], temperatures[i+1]
            w1 = (t2 - target_temp) / (t2 - t1)
            w2 = 1 - w1
            return [w1 * rv_precision[t1][j] + w2 * rv_precision[t2][j] for j in range(len(rv_precision[t1]))]

def calculate_custom_rvp(precisions: List[float], wavelengths: List[float], lam_min: float, lam_max: float) -> float:
    custom_rvp = []
    for i, wavelength in enumerate(wavelengths):
        if lam_min <= wavelength <= lam_max:
            custom_rvp.append(precisions[i])

    if not custom_rvp:
        raise ValueError("No values in selected wavelength range!")

    sum_inv_square = sum(1 / (x*x) for x in custom_rvp if not math.isnan(x))
    return 1 / math.sqrt(sum_inv_square)

def calculate_rv_precision(values: Dict[str, float], data_file: str) -> float:
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    wavelengths = get_wavelengths(all_rows)
    temperatures = get_temperatures(list(all_rows[0].keys()))
    rv_precision = get_rv_precision(all_rows, temperatures)

    interpolated_precision = interpolate_rv_precision(rv_precision, temperatures, values['Temp'])

    # Normalize with SNR and spectral Resolution
    snr = 100 * (values['MiDia'] / 4) * math.pow(10, 0.2 * (9 - values['Vmag'])) * math.sqrt(values['TObs'] / 10)
    sp_res_factor = math.pow(4000 / values['SpRes'], 1.5)

    normalized_precision = [p * (100 / snr) * sp_res_factor for p in interpolated_precision]

    return calculate_custom_rvp(normalized_precision, wavelengths['Lambda'], values['Lam_min'], values['Lam_max'])

if __name__ == "__main__":
    values = get_manual_values()
    data_file = "dataVmag.csv"  # Ensure this file is in the same directory as your script
    
    try:
        rv_precision = calculate_rv_precision(values, data_file)
        print(f"Calculated RV Precision: {rv_precision:.2f} m/s")
    except Exception as e:
        print(f"Error: {e}")