import math
import csv
import numpy as np
from typing import Dict, List


# Define the standard values
standVals = {
    "snrVmag": 100,
    "miDia": 4.0,
    "Vmag": 9.0,
    "tObs": 10.0,
    "spRes": 80000
}


def get_manual_values(Temp: float = 4200.0, Vmag: float = 7.0) -> Dict[str, float]:
    return {
        "Temp": Temp,    # Temperature in K
        "Vmag": Vmag,    # V magnitude
        "MiDia": 2.5,    # Mirror Diameter in m
        "SpRes": 110000, # Spectral Resolution
        "TObs": 20.0,    # Observation Time in min
        "Lam_min": 380,  # Minimum wavelength in nm;
        "Lam_max": 670   # Maximum wavelength in nm; 
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


def get_temperatures(header: List[str]) -> List[int]:
    return [int(temp) for temp in header if temp not in ['Lam_min', 'Lam_max']]


def get_rv_precision(all_rows: List[Dict], temperatures: List[int]) -> Dict[int, List[float]]:
    rv_precision = {temp: [] for temp in temperatures}
    for row in all_rows:
        for temp in temperatures:
            rv_precision[temp].append(float(row[str(temp)]))
    return rv_precision


def interpolate_temperature(rv_precision, temperatures, temp):
    if temp <= min(temperatures):
        return rv_precision[min(temperatures)]
    elif temp >= max(temperatures):
        return rv_precision[max(temperatures)]
    else:
        lower_temp = max(t for t in temperatures if t <= temp)
        upper_temp = min(t for t in temperatures if t >= temp)
        lower_rv = rv_precision[lower_temp]
        upper_rv = rv_precision[upper_temp]
        return [np.interp(temp, [lower_temp, upper_temp], [lower_rv[i], upper_rv[i]]) 
                for i in range(len(lower_rv))]


def get_custom_rvp(values, lam_min, lam_max):
    precisions = values["RVPrecision"]
    wavelengths = values["Wavelengths"]["Lambda"]
    
    # Use the provided Lam_min and Lam_max
    rv_window = [lam_min, lam_max]

    # Ensure the window is within the available wavelength range
    rv_window[0] = max(rv_window[0], min(wavelengths))
    rv_window[1] = min(rv_window[1], max(wavelengths))

    # Check if precisions is a single value or an iterable
    if np.isscalar(precisions):
        custom_rvp = [precisions for w in wavelengths if rv_window[0] <= w <= rv_window[1]]
    else:
        custom_rvp = [p for w, p in zip(wavelengths, precisions) if rv_window[0] <= w <= rv_window[1]]

    if not custom_rvp:
        print("No value lies in selected wavelength range!")
        return None

    sum_inv_square = sum(1 / (rv**2) for rv in custom_rvp if not math.isnan(rv))
    return 1 / math.sqrt(sum_inv_square) if sum_inv_square > 0 else None


def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)    


def rv_precision(all_rows, values):
    mag = "Vmag"  # Assuming we're using V magnitude

    temperatures = get_temperatures(all_rows[0].keys())
    wavelengths = get_wavelengths(all_rows)
    rv_precision_data = get_rv_precision(all_rows, temperatures)

    values["SNR" + mag] = standVals["snrVmag"] / values["SNRVmag"]
    values["SpRes"] = (standVals["spRes"] / values["SpRes"]) ** 1.5
    values["Wavelengths"] = wavelengths

    rv_precision_values = interpolate_temperature(rv_precision_data, temperatures, values["Temp"])
    
    if rv_precision_values[0] == -1:
        print(f"Temperature {values['Temp']} is out of range.")
        values["RVPrecision"] = None
    else:
        values["RVPrecision"] = [rv * values["SNR" + mag] * values["SpRes"] for rv in rv_precision_values]

    values["Wavelengths"] = wavelengths
    custom_rv_precision = get_custom_rvp(values, values["Lam_min"], values["Lam_max"])

    return values, custom_rv_precision


def calculate_rv_precision(Temp: float = None, Vmag: float = None):
    values = get_manual_values()
    if Temp is not None:
        values["Temp"] = Temp
    if Vmag is not None:
        values["Vmag"] = Vmag
    
    values["SNRVmag"] = standVals["snrVmag"] * (values["MiDia"] / standVals["miDia"]) * \
                        10 ** (0.2 * (standVals["Vmag"] - values["Vmag"])) * \
                        math.sqrt(values["TObs"] / standVals["tObs"])

    all_rows = read_csv("../data/dataVmag.csv")
    result, custom_rv_precision = rv_precision(all_rows, values)

    return result, custom_rv_precision    


# Run the calculation
if __name__ == "__main__":
    result, custom_rv_precision = calculate_rv_precision()
    print(f"RV Precision: {result['RVPrecision']}")
    print(f"Custom RV Precision: {custom_rv_precision}")
