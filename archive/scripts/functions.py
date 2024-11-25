import math
import csv
import requests
from io import StringIO

# Global variables
magnitude = ""
magNaN = ""
inputSelection = ""
Wavelengths = {}
Temperatures = []
Tmin = 0
Tmax = 0
RVPrecision = {}
Values = {}
xvals = []
xerrs = []
yvals = []
optional = []
standVals = {}  # You need to define this dictionary with standard values

def get_manual_values():
    # This function would need to be adapted to get values from a GUI or command line
    return {
        "Vmag": [float(input("Enter Vmag: "))],
        "Jmag": [float(input("Enter Jmag: "))],
        "Temp": [float(input("Enter Temperature: "))],
        "SpRes": float(input("Enter Spectral Resolution: ")),
        "SNRVmag": float(input("Enter SNR Vmag: ")),
        "SNRJmag": float(input("Enter SNR Jmag: ")),
        "MiDia": float(input("Enter Mirror Diameter: ")),
        "TObs": float(input("Enter Observation Time: ")),
    }

def to_snr():
    global Values
    new_values = get_manual_values()
    if magNaN != "jmag" and magnitude == "vmag":
        mag = "Vmag"
    elif magNaN != "vmag" and magnitude == "jmag":
        mag = "Jmag"
    
    new_values["SNR"+mag] = standVals["snr"+mag] * \
        (new_values["MiDia"]/standVals["miDia"]) * \
        math.pow(10, 0.2 * (standVals[magnitude]-new_values[mag][0])) * \
        math.sqrt(new_values["TObs"]/standVals["tObs"])
    Values = new_values
    return Values["SNR"+mag]

def calc_new_tobs():
    global Values
    new_values = get_manual_values()
    mag = "Vmag" if magnitude == "vmag" else "Jmag"
    
    new_values["TObs"] = standVals["tObs"] * \
        math.pow(standVals["miDia"]/new_values["MiDia"], 2) * \
        math.pow(10, -0.4 * (standVals[magnitude]-new_values[mag][0])) * \
        math.pow(new_values["SNR"+mag]/standVals["snr"+mag], 2)
    Values = new_values
    return Values["TObs"]

def update_slave(write_to_html):
    # This function would need to be adapted for Python GUI or command line
    pass

def coloring(x):
    # This function would need to be adapted for Python GUI
    pass

def query_variable(variable, write_to_html):
    if variable == "SNR":
        new_var = to_snr()
        if write_to_html:
            # This part would need to be adapted for Python GUI or command line output
            print(f"New SNR: {new_var:.2f}")
    elif variable == "TObs":
        new_var = calc_new_tobs()
        if write_to_html:
            print(f"New Observation Time: {new_var:.2f}")

def prepare_download(values, file_ending=""):
    delim = ","
    output = "Id,Coordinates,Spectral Type,Temperature,Vmag,Jmag,"
    
    for i in range(len(Wavelengths["Lambda"])):
        output += f"RV_Prec({Wavelengths['Lambda'][i]} +- {Wavelengths['Lambda_err'][i]}){delim}"
    
    rv_window = [float(input("Enter Lam_min: ") or Wavelengths["Lambda"][0]),
                 float(input("Enter Lam_max: ") or Wavelengths["Lambda"][-1])]
    output += f"RV_Prec({rv_window[0]} - {rv_window[1]})\n"

    for i in range(len(values["SpT"])):
        output += f"{values['Id'][i]}{delim}{values['Coord'][i]}{delim}"
        output += f"{values['SpT'][i]}{delim}{values['Temp'][i]}{delim}"
        output += f"{values['Vmag'][i]}{delim}{values['Jmag'][i]}{delim}"
        for j in range(len(values["RVPrecision"][i])):
            output += f"{values['RVPrecision'][i][j]}{delim}"
        output += f"{get_custom_rvp(values, i)}\n"
    
    return output

def spectral_to_temperature(classification, id, coord, spt):
    simple_spt = simplify_spt(spt)
    for classirow in classification:
        if simplify_spt(classirow['Stellar Type']).find(simple_spt) != -1:
            return float(classirow['Temp K'])
    print(f"Spectral Type of {id} ({coord}) was not understood or found!")
    print(f"Simbad returned as its Spectral Type {spt}")
    raise ValueError("ERROR in fetching spectral type from simbad!")

def get_wavelengths(all_rows):
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

def get_temperatures(header):
    return [float(temp) for temp in header]

def get_rv_precision(all_rows, temperatures):
    rv_precision = {temp: [] for temp in temperatures}
    for row in all_rows:
        for temp in temperatures:
            rv_precision[temp].append(float(row[str(temp)]))
    return rv_precision

def get_optional(from_listener):
    # This function would need to be adapted for Python GUI or command line
    pass

def get_custom_rvp(values, k=0):
    custom_rvp = []
    precisions = values["RVPrecision"][k]
    wavelengths = Wavelengths["Lambda"]
    rv_window = optional or [float('nan'), float('nan')]

    if math.isnan(rv_window[0]) or rv_window[1] < min(wavelengths):
        rv_window[0] = min(wavelengths)
    if math.isnan(rv_window[1]) or rv_window[1] > max(wavelengths):
        rv_window[1] = max(wavelengths)

    for i, wavelength in enumerate(wavelengths):
        if rv_window[0] <= wavelength <= rv_window[1]:
            custom_rvp.append(precisions[i])

    if not custom_rvp:
        print("No value lays in selected wavelength range!")
        return None

    sum_inv_square = sum(1 / (x*x) for x in custom_rvp if not math.isnan(x))
    return 1 / math.sqrt(sum_inv_square)

def rv_precision(all_rows, values, write_to_html):
    global Wavelengths, Temperatures, Tmin, Tmax, RVPrecision

    mag = "Jmag" if magnitude == "jmag" else "Vmag"

    header = list(all_rows[0].keys())
    header = header[:-2]  # Remove Lam_max and Lam_min

    Wavelengths = get_wavelengths(all_rows)
    Temperatures = get_temperatures(header)
    Tmin, Tmax = min(Temperatures), max(Temperatures)
    RVPrecision = get_rv_precision(all_rows, Temperatures)

    values["RVPrecision"] = [[] for _ in range(len(values['Vmag']))]
    values["SNR"+mag] = standVals["snr"+mag] / values["SNR"+mag]
    values["SpRes"] = math.pow(standVals["spRes"] / values["SpRes"], 1.5)

    for i in range(len(values[mag])):
        values["RVPrecision"][i] = check_input(RVPrecision, Temperatures, values["Temp"][i])

        if values["RVPrecision"][i][0] == -1:
            if write_to_html:
                if values["Temp"][i] < Tmin:
                    print(f"(*)temperature: {values['Temp'][i]} smaller than {Tmin}")
                elif values["Temp"][i] > Tmax:
                    print(f"(*)temperature: {values['Temp'][i]} bigger than {Tmax}")
            else:
                if values["Temp"][i] < Tmin:
                    print(f"(*){values['Id'][i]} ({values['Coord'][i]}) temperature: {values['Temp'][i]} smaller than {Tmin}")
                elif values["Temp"][i] > Tmax:
                    print(f"(*){values['Id'][i]} ({values['Coord'][i]}) temperature: {values['Temp'][i]} bigger than {Tmax}")
            continue

        for j in range(len(values["RVPrecision"][i])):
            values["RVPrecision"][i][j] *= values["SNR"+mag] * values["SpRes"]

    if write_to_html and -1 in values["RVPrecision"][0]:
        return

    if write_to_html:
        get_optional(False)
        custom_rv_precision = get_custom_rvp(values)
        update_table(values, custom_rv_precision)
        global xvals, xerrs, yvals
        xvals = Wavelengths["Lambda"]
        xerrs = Wavelengths["Lambda_err"]
        yvals = values['RVPrecision'][0]
        make_trace(xvals, xerrs, yvals, custom_rv_precision, *optional)
    else:
        output = prepare_download(values, mag)
        with open(f"output{mag}.csv", "w") as f:
            f.write(output)
        print(output)

# The rest of the functions (analyse_simbad_query, query_simbad, query_batch, etc.) 
# would need to be translated similarly, adapting web-specific parts to Python equivalents.

# Main execution
if __name__ == "__main__":
    # This would be the entry point of your Python script
    # You'd need to implement the main logic here, possibly using a command-line interface
    # or a Python GUI framework to replicate the web interface functionality
    pass
