import math
from typing import Dict, List

import requests
import pandas as pd
from tkinter import filedialog, Tk
import json

# -----------------------------
# from helpers.js
# -----------------------------

import numpy as np

def nanmin(x):
    return np.nanmin(x)

def argnanmin(x):
    return np.nanargmin(x)

def simplify_spt(spt):
    int_spt = spt.replace("dM", "M")[:2]
    for letter in ["IV", "V", "III", "II", "I"]:
        if letter in spt:
            return int_spt + letter
    return int_spt + "V"

def find_temperature(temp_array, temperature):
    diff_old = temp_array[0] - temperature
    for i in range(1, len(temp_array)):
        diff = temp_array[i] - temperature
        if diff_old < 0 and diff < 0:
            diff_old = diff
        elif diff_old < 0 and diff > 0:
            return [temp_array[i-1], temp_array[i]]
    raise ValueError(f"ERROR in finding temperature interval! Given Temperature: {temperature}")

def lin_interpol(temp, temp_int, values_min, values_max):
    return [values_min[i] + (temp - temp_int[0]) * (values_max[i] - values_min[i]) / (temp_int[1] - temp_int[0]) for i in range(len(values_min))]

def check_input(dict_values, temperatures, given_temp):
    if given_temp < min(temperatures) or given_temp > max(temperatures):
        return [-1] * len(temperatures)
    
    if given_temp in temperatures:
        return dict_values[given_temp]
    
    temp_interval = find_temperature(temperatures, given_temp)
    values = lin_interpol(given_temp, temp_interval, dict_values[temp_interval[0]], dict_values[temp_interval[1]])
    
    if all(np.isnan(values)):
        if magnitude == "jmag":
            magnitude = "vmag"
            document.getElementById("jmag").style.display = "none"
            mag_nan = "vmag"
            input_function()
        elif magnitude == "vmag":
            magnitude = "jmag"
            document.getElementById("vmag").style.display = "none"
            mag_nan = "jmag"
            input_function()
    
    return values

def update_table(values, custom_rv_precision):
    table = document.getElementById("table")
    
    rv_window = optional
    if np.isnan(rv_window[0]):
        rv_window[0] = min(Wavelengths["Lambda"])
    if np.isnan(rv_window[1]):
        rv_window[1] = max(Wavelengths["Lambda"])
    
    for j in range(table.rows.length - 1, 0, -1):
        table.deleteRow(1)
    
    for j in range(len(Wavelengths["Lambda"])):
        row = table.insertRow(table.rows.length)
        row.insertCell(0).innerHTML = round(Wavelengths["Lambda"][j] - Wavelengths["Lambda_err"][j], 2)
        row.insertCell(1).innerHTML = round(Wavelengths["Lambda"][j] + Wavelengths["Lambda_err"][j], 2)
        row.insertCell(2).innerHTML = round(values["RVPrecision"][0][j], 2)
    
    if not np.isnan(rv_window[0]) and not np.isnan(rv_window[1]):
        row = table.insertRow(table.rows.length)
        row.style['background-color'] = custom_range_color
        row.insertCell(0).innerHTML = round(rv_window[0], 2)
        row.insertCell(1).innerHTML = round(rv_window[1], 2)
        row.insertCell(2).innerHTML = round(custom_rv_precision, 2)
    
    index_of_min_value = argnanmin(values["RVPrecision"][0])
    if index_of_min_value + 1:
        table.rows[index_of_min_value + 1].style.background = coloring_color
    
    table.style.visibility = "visible"

def download(filename, text):
    element = document.getElementById("outputfile")
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text))
    # element.setAttribute('download', filename)
    # element.style.display = 'none'
    # document.body.appendChild(element)
    # element.click()
    # document.body.removeChild(element)

# Note: The commented-out function queryTObs is not converted as it is marked as TODO and not fully implemented.



# -----------------------------
# from functions.js
# -----------------------------

# TODO urgent: HD 5 gibt nen Fehler weil G2/3V nicht verstanden wird!
# TODO: Add t_obs "min", "sec", "hour" switch!
# TODO: simbad query: changing SpRes leads to new simbad call!
# TODO urgent: Einzelne Variablen verändern und <Enter> geht. Mehrere verändern und Button drücken -> Fehler! Rechne erst entsprechende
# Variablen um (Schritt-für-Schritt) und verwende dann die Werte für RV Precision.

def get_manual_values() -> Dict:
    """
    Returns a dictionary with values from the input fields on the website.
    """
    return {
        "Vmag": [float(document.getElementById("vmag").value)],
        "Jmag": [float(document.getElementById("jmag").value)],
        "Temp": [float(document.getElementById("temp").value)],
        "SpRes": float(document.getElementById("SpRes").value),
        "SNRVmag": float(document.getElementById("snrVmag").value),
        "SNRJmag": float(document.getElementById("snrJmag").value),
        "MiDia": float(document.getElementById("MiDia").value),
        "TObs": float(document.getElementById("tObs").value),
    }

def to_snr() -> float:
    """
    Returns new SNR from Vmag.
    """
    new_values = get_manual_values()
    if magNaN != "jmag" and magnitude == "vmag":
        mag = "Vmag"
    elif magNaN != "vmag" and magnitude == "jmag":
        mag = "Jmag"
    
    # SNR = 100 * 10^[0.2(9mag-V)] * (MiDia/4m) * (TObs / 10 min)^0.5
    new_values[f"SNR{mag}"] = stand_vals[f"snr{mag}"] * (new_values["MiDia"] / stand_vals["miDia"]) * \
        math.pow(10, 0.2 * (stand_vals[magnitude] - new_values[mag])) * \
        math.sqrt(new_values["TObs"] / stand_vals["tObs"])
    
    values = new_values
    return values[f"SNR{mag}"]

def calc_new_t_obs() -> float:
    """
    Returns new observing time and updates the Values object.
    """
    new_values = get_manual_values()
    if magnitude == "vmag":
        mag = "Vmag"
    else:
        mag = "Jmag"

    # m = -2.5 log10(F); F = 10^{-0.4m}
    # tobs = 10 min * (10^[-0.2(9mag-V)])^2 * (4m/MiDia)^2 * (SNR/100)^2
    new_values["TObs"] = stand_vals["tObs"] * \
                         math.pow(stand_vals["miDia"] / new_values["MiDia"], 2) * \
                         math.pow(10, -0.4 * (stand_vals[magnitude] - new_values[mag])) * \
                         math.pow(new_values[f"SNR{mag}"] / stand_vals[f"snr{mag}"], 2)
    values = new_values
    return values["TObs"]

def update_slave(write_to_html: bool):
    if document.querySelector(".master > input").name == 'snr':
        query_variable('TObs', write_to_html)
    elif document.querySelector(".master > input").name == 'observingtime':
        query_variable('SNR', write_to_html)

def coloring(x):
    x.style.backgroundColor = coloring_color
    setTimeout(lambda: x.style.backgroundColor = "unset", coloring_time)  # reset color after 'coloring_time' seconds

def query_variable(variable: str, write_to_html: bool):
    """
    Checks for new tObs or SNR and updates input field with color.
    """
    if variable == "SNR":
        new_var = to_snr()
        if write_to_html:
            if magNaN != "vmag" and magnitude == "jmag":
                coloring(document.getElementById("inputTableSnrJmag"))
                document.getElementById("snrJmag").value = f"{new_var:.2f}"
                document.getElementById("snrJmag").parentElement.style.display = ""
                document.getElementById("snrVmag").parentElement.style.display = "none"
            elif magNaN != "jmag" and magnitude == "vmag":
                coloring(document.getElementById("inputTableSnrVmag"))
                document.getElementById("snrVmag").value = f"{new_var:.2f}"
                document.getElementById("snrVmag").parentElement.style.display = ""
                document.getElementById("snrJmag").parentElement.style.display = "none"
    elif variable == "TObs":
        new_var = calc_new_t_obs()
        if write_to_html:
            coloring(document.getElementById("tObs").parentElement)
            document.getElementById("tObs").value = f"{new_var:.2f}"

def prepare_download(values: Dict, file_ending: str = ""):
    delim = ","
    output = "Id,Coordinates,Spectral Type,Temperature,"
    # add to header the wavelengths provided
    for i in range(len(Wavelengths["Lambda"])):
        output += f"RV_Prec({Wavelengths['Lambda'][i]} +- {Wavelengths['Lambda_err'][i]}){delim}"
    # add full wavelength range
    output += f"RV_Prec({Wavelengths['Lambda'][0]} - {Wavelengths['Lambda'][-1]})\n"

    for i in range(len(values["SpT"])):
        output += f"{values['Id'][i]}{delim}{values['Coord'][i]}{delim}{values['SpT'][i]}{delim}{values['Temp'][i]}{delim}"
        for j in range(len(values["RVPrecision"][i])):
            output += f"{values['RVPrecision'][i][j]}{delim}"
        output += f"{get_custom_rvp(values, i)}\n"

    download(f"output{file_ending}.csv", output)

def spectral_to_temperature(classification: List[Dict], id: str, coord: str, spT: str, write_to_html: bool) -> float:
    simple_spT = simplify_spT(spT)
    for classirow in classification:
        if simple_spT in simplify_spT(classirow['Stellar Type']):
            return float(classirow['Temp K'])
    document.getElementById("simbad_err").innerHTML += f"<br>Spectral Type of <br>{id} ({coord})<br>was not understood resp. found!"
    document.getElementById("simbad_err").innerHTML += f"<br>Simbad returned as its Spectral Type {spT}"
    raise ValueError("ERROR in fetching spectral type from simbad!")

def get_wavelengths(all_rows: List[Dict]) -> Dict:
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
    """
    Get temperatures from given header of file.
    """
    return [float(h) for h in header]

def get_rv_precision(all_rows: List[Dict], temperatures: List[float]) -> Dict:
    """
    Get RV Precision from file. Save as dictionary with Temps as keys.
    """
    rv_precision = {temp: [] for temp in temperatures}
    for row in all_rows:
        for temp in temperatures:
            rv_precision[temp].append(float(row[str(temp)]))
    return rv_precision

def get_optional(from_listener: bool):
    lam_min_cut = float(document.getElementById("Lam_min").value)
    lam_max_cut = float(document.getElementById("Lam_max").value)
    if lam_min_cut > lam_max_cut:
        document.getElementById("simbad_err").innerHTML = "λ<sub>min</sub> bigger than λ<sub>max</sub>!"
        return
    if lam_min_cut == lam_max_cut:
        document.getElementById("simbad_err").innerHTML = "λ<sub>min</sub> and λ<sub>max</sub> are equal but should not!"
        return
    document.getElementById("simbad_err").innerHTML = ""
    optional = [lam_min_cut, lam_max_cut]
    if from_listener and xvals:
        optional[0] = max(optional[0], min(Wavelengths["Lambda"]))
        optional[1] = min(optional[1], max(Wavelengths["Lambda"]))
        make_trace(xvals, xerrs, yvals)
        update_table(values, custom_rv_precision)

def get_custom_rvp(values: Dict, k: int = 0) -> float:
    custom_rvp = []
    precisions = values["RVPrecision"][k]
    wavelengths = Wavelengths["Lambda"]
    rv_window = optional or [float('nan'), float('nan')]

    if math.isnan(rv_window[0]) or rv_window[1] < min(wavelengths):
        rv_window[0] = min(wavelengths)
    if math.isnan(rv_window[1]) or rv_window[1] > max(wavelengths):
        rv_window[1] = max(wavelengths)

    for i in range(len(wavelengths)):
        if wavelengths[i] >= rv_window[0] and wavelengths[i] <= rv_window[1]:
            custom_rvp.append(precisions[i])
    
    if not custom_rvp:
        document.getElementById("simbad_err").innerHTML = "No value lays in selected wavelength range!"
        return

    sum_inv_sq = sum(1 / (x * x) for x in custom_rvp if not math.isnan(x))
    return 1 / math.sqrt(sum_inv_sq)

def rv_precision(all_rows: List[Dict], values: Dict, write_to_html: bool):
    """
    Get the RV Precision and prepare final dataset for plot/download.
    """
    mag = "Jmag" if magnitude == "jmag" else "Vmag"

    header = list(all_rows[0].keys())
    header.pop()
    header.pop()

    global Wavelengths, Temperatures, Tmin, Tmax, RVPrecision
    Wavelengths = get_wavelengths(all_rows)
    Temperatures = get_temperatures(header)
    Tmin = min(Temperatures)
    Tmax = max(Temperatures)
    RVPrecision = get_rv_precision(all_rows, Temperatures)

    values["RVPrecision"] = [[] for _ in values['Vmag']]
    values[f"SNR{mag}"] = stand_vals[f"snr{mag}"] / values[f"SNR{mag}"]
    values["SpRes"] = math.pow(stand_vals["spRes"] / values["SpRes"], 1.5)

    for i in range(len(values[mag])):
        values["RVPrecision"][i] = check_input(RVPrecision, Temperatures, values["Temp"][i])

        if values["RVPrecision"][i][0] == -1:
            if write_to_html and values["Temp"][i] < Tmin:
                document.getElementById("simbad_err").innerHTML += f"<br>(*)temperature: {values['Temp'][i]} smaller than {Tmin}"
            elif not write_to_html and values["Temp"][i] < Tmin:
                document.getElementById("simbad_err").innerHTML += f"<br>(*){values['Id'][i]} ({values['Coord'][i]}) temperature: {values['Temp'][i]} smaller than {Tmin}"
            elif write_to_html and values["Temp"][i] > Tmax:
                document.getElementById("simbad_err").innerHTML += f"<br>(*)temperature: {values['Temp'][i]} bigger than {Tmax}"
            elif not write_to_html and values["Temp"][i] > Tmax:
                document.getElementById("simbad_err").innerHTML += f"<br>(*){values['Id'][i]} ({values['Coord'][i]}) temperature: {values['Temp'][i]} bigger than {Tmax}"
            continue
        
        for j in range(len(values["RVPrecision"][i])):
            values["RVPrecision"][i][j] *= values[f"SNR{mag}"] * values["SpRes"]

    if write_to_html and -1 in values["RVPrecision"][0]:
        return
    else:
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
            prepare_download(values, mag)

def analyse_simbad_query(url_rows: List[Dict], classification_rows: List[Dict], write_to_html: bool):
    header = list(url_rows[0].keys())

    variables = {
        "SpT": [None] * (len(url_rows) - 1),
        "Vmag": [None] * (len(url_rows) - 1),
        "Jmag": [None] * (len(url_rows) - 1),
        "Id": [None] * (len(url_rows) - 1),
        "Coord": [None] * (len(url_rows) - 1),
        "Temp": [None] * (len(url_rows) - 1),
        "SpRes": float(document.getElementById("SpRes").value),
        "SNRVmag": float(document.getElementById("snrVmag").value),
        "SNRJmag": float(document.getElementById("snrJmag").value),
        "MiDia": float(document.getElementById("MiDia").value),
        "TObs": float(document.getElementById("tObs").value),
    }

    for i, row in enumerate(url_rows):
        variables["SpT"][i] = row["SpT"].split(' ')[0]
        variables["Vmag"][i] = float(row["Vmag"])
        variables["Jmag"][i] = float(row["Jmag"])
        variables["Id"][i] = row["Name"]
        variables["Coord"][i] = row["Coo"]
        variables["Temp"][i] = spectral_to_temperature(classification_rows, variables["Id"][i], variables["Coord"][i], variables["SpT"][i], write_to_html)

        if write_to_html:
            if magNaN != "vmag" and not math.isnan(variables["Jmag"][i]) and (math.isnan(variables["Vmag"][i]) or variables["Temp"][i] <= 4000):
                document.getElementById("Magnitude-jband").selected = "true"
                magnitude = "jmag"
                document.getElementById("jmag").style.display = ""
                document.getElementById("vmag").style.display = "none"
                document.getElementById("snrJmag").parentElement.style.display = ""
                document.getElementById("snrVmag").parentElement.style.display = "none"
                if 'snr' == document.querySelector('.master > input').name:
                    togglemaster(document.getElementById("snrJmag"))
            else:
                document.getElementById("Magnitude-vband").selected = "true"
                magnitude = "vmag"
                document.getElementById("jmag").style.display = "none"
                document.getElementById("vmag").style.display = ""
                document.getElementById("snrVmag").parentElement.style.display = ""
                document.getElementById("snrJmag").parentElement.style.display = "none"
                if 'snr' == document.querySelector('.master > input').name:
                    togglemaster(document.getElementById("snrVmag"))

    nan2emptystr = lambda x: "" if math.isnan(x) else x
    is_new_temp = document.getElementById("temp").value != nan2emptystr(variables["Temp"][0])
    is_new_vmag = document.getElementById("vmag").value != nan2emptystr(variables["Vmag"][0])
    is_new_jmag = document.getElementById("jmag").value != nan2emptystr(variables["Jmag"][0])
    document.getElementById("temp").value = variables["Temp"][0]
    document.getElementById("vmag").value = variables["Vmag"][0]
    document.getElementById("jmag").value = variables["Jmag"][0]
    update_slave(write_to_html)

    if is_new_temp:
        coloring(document.getElementById("temp").parentElement)
    if is_new_vmag:
        coloring(document.getElementById("vmag").parentElement)
    if is_new_jmag:
        coloring(document.getElementById("jmag").parentElement)

    global magNaN
    magNaN = ""
    if magnitude == "vmag":
        Plotly.d3.csv("dataVmag.csv", lambda all_rows: rv_precision(all_rows, variables, write_to_html))
    elif magnitude == "jmag":
        Plotly.d3.csv("dataJmag.csv", lambda all_rows: rv_precision(all_rows, variables, write_to_html))


def query_simbad(url, write_to_html):
    base_url = 'https://simbad.u-strasbg.fr/simbad/sim-script?script='
    output = 'output%20console=off%20script=off%20error=off%0D'
    format = 'format%20object%20%22%25IDLIST(1),%2527COO(A%20D),%25SP,%25FLUXLIST(V%3BF),%25FLUXLIST(J%3BF)%22'
    header = 'echodata%20Name,Coo,SpT,Vmag,Jmag'
    full_url = f"{base_url}{output}%0A{format}%0A{header}%0A{url}"
    
    print("Simbad Query:", full_url)
    
    response = requests.get(full_url)
    if response.status_code != 200:
        print("Error fetching data from Simbad")
        return
    
    data = response.text
    rows = data.split("\n")
    if not rows or "error" in rows[0]:
        print("Given object(s) not found by Simbad!")
        return
    
    simbad_data = pd.read_csv(pd.compat.StringIO(data))
    classi_data = pd.read_csv("classification.csv")
    analyse_simbad_query(simbad_data, classi_data, write_to_html)

def file_upload():
    Tk().withdraw()
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    write_to_html = len(lines) < 2
    query_str = "%0A".join(line.strip() for line in lines)
    query_simbad(query_str, write_to_html)

def key_down(event):
    if event.key == 'Enter':
        event.preventDefault()
        if event.target.id in ["snrVmag", "snrJmag", "MiDia", "vmag", "jmag", "tObs"]:
            update_slave(True)
        
        query_button.style.backgroundColor = '#666'
        setTimeout(lambda: query_button.style.backgroundColor = "", 100)
        input_function()

def simbad_field_input():
    co_query = document.getElementById("co_query")
    id_query = document.getElementById("id_query")
    query = ""
    if id_query.checked:
        query += 'query%20id%20'
    elif co_query.checked:
        query += 'query%20coo%20'
    else:
        document.getElementById("simbad_err").innerHTML = "Neither 'identifier' nor 'coordinates' selected! Select one and repeat the query!"
        return
    
    document.getElementById("simbad_err").innerHTML = ""
    simbad = document.getElementById("simbadField").value
    query += simbad
    query_simbad(query, True)

def which_mag_band():
    selection = document.getElementById("Magnitude")
    v_select = document.getElementById("vmag")
    j_select = document.getElementById("jmag")
    snr_v_select = document.getElementById("inputTableSnrVmag")
    snr_j_select = document.getElementById("inputTableSnrJmag")

    update_slave(True)

    if selection[0].selected:
        v_select.style.display = ""
        snr_v_select.style.display = ""
        j_select.style.display = "none"
        snr_j_select.style.display = "none"
        magnitude = "vmag"
        if xvals:
            mag_nan = "vmag"
            input_function()
    elif selection[1].selected:
        v_select.style.display = "none"
        snr_v_select.style.display = "none"
        j_select.style.display = ""
        snr_j_select.style.display = ""
        magnitude = "jmag"
        if xvals:
            mag_nan = "jmag"
            input_function()

def which_radio():
    file_input = document.getElementById("fileInput")
    manually = document.getElementById("setManually")
    id_query = document.getElementById("id_query")
    co_query = document.getElementById("co_query")

    document.getElementById("temp").readOnly = not manually.checked
    document.getElementById("vmag").readOnly = not manually.checked
    document.getElementById("jmag").readOnly = not manually.checked
    document.getElementById("simbadField").readOnly = not (id_query.checked or co_query.checked)
    simbad_field.placeholder = "Simbad coords" if co_query.checked else "Simbad resolver"

    if file_input.checked:
        input_selection = "fileInput"
    elif id_query.checked or co_query.checked:
        input_selection = "idcoQuery"
    elif manually.checked:
        input_selection = "manual"

def input_function():
    update_slave(True)

    document.getElementById("simbad_err").innerHTML = ""
    if not input_selection:
        document.getElementById("simbad_err").innerHTML = "No type of query selected! Select one from above and repeat the query!"
    elif input_selection == "idcoQuery":
        simbad_field_input()
    elif input_selection == "fileInput":
        file_upload()
        coloring(outputfile)
    elif input_selection == "manual":
        variables = get_manual_values()
        if magnitude == "jmag":
            data = pd.read_csv("dataJmag.csv")
            rv_precision(data, variables, True)
        else:
            data = pd.read_csv("dataVmag.csv")
            rv_precision(data, variables, True)

def toggle_master(elem):
    x = document.querySelector('.master > input')
    if x:
        x.parentElement.classList.remove('master')
    (elem or event.target).parentElement.classList.add('master')

# Event listeners
document.getElementById("Magnitude").addEventListener("change", which_mag_band)
document.getElementById("fileInput").addEventListener("change", which_radio)
document.getElementById("id_query").addEventListener("change", which_radio)
document.getElementById("co_query").addEventListener("change", which_radio)
document.getElementById("setManually").addEventListener("change", which_radio)

document.getElementById("temp").addEventListener("keydown", key_down)
document.getElementById("vmag").addEventListener("keydown", key_down)
document.getElementById("jmag").addEventListener("keydown", key_down)
document.getElementById("MiDia").addEventListener("keydown", key_down)
document.getElementById("SpRes").addEventListener("keydown", key_down)
document.getElementById("snrVmag").addEventListener("keydown", key_down)
document.getElementById("snrJmag").addEventListener("keydown", key_down)
document.getElementById("tObs").addEventListener("keydown", key_down)
document.getElementById("Lam_min").addEventListener("keydown", key_down)
document.getElementById("Lam_max").addEventListener("keydown", key_down)

document.getElementById("queryButton").addEventListener("click", input_function)
document.getElementById("simbadField").addEventListener("keydown", key_down)

which_radio()
input_function()

