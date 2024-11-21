import time
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import numpy as np
import warnings
from astropy.units import UnitsWarning
from requests.exceptions import ConnectionError
import logging
import pandas as pd

# Read the CSV file as a DataFrame
classification_df = pd.read_csv("../data/MathiasZechmeister/classification.csv")

# Configure logging to write to a file
logging.basicConfig(
    filename='stellar_properties_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress specific warning
warnings.simplefilter('ignore', category=UnitsWarning)

def get_simbad_info_with_retry(source_id, retries=3, delay=5):
    for attempt in range(retries):
        try:
            result_table = custom_simbad.query_object(f"Gaia DR3 {source_id}")
            if result_table is not None:
                ids = result_table['IDS'][0].split('|')
                hd_numbers = [id.strip() for id in ids if id.startswith('HD')]
                gj_numbers = [id.strip() for id in ids if id.startswith('GJ')]
                hip_numbers = [id.strip() for id in ids if id.startswith('HIP')]
                object_type = result_table['OTYPE'][0]
                return {
                    'HD Number': ', '.join(hd_numbers) if hd_numbers else None,
                    'GJ Number': ', '.join(gj_numbers) if gj_numbers else None,
                    'HIP Number': ', '.join(hip_numbers) if hip_numbers else None,
                    'Object Type': object_type
                }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None

def clean_hd_number(hd_string):
    """Extract the first HD number from a given string."""
    if pd.isna(hd_string) or not hd_string.startswith("HD"):
        return None
    # Split by comma and take the first part, then remove "HD" and whitespace
    return hd_string.split(',')[0].replace("HD", "").strip()

def extract_mass(hd_number):
    file_path = '../data/Catalogue_V_117A_table1.txt'
    with open(file_path, 'r') as file:
        for line in file:
            # Extract the HD number from the line
            line_hd_number = line[7:18].strip()
            # Check if the line contains the desired HD number
            if line_hd_number == f"HD {hd_number}":
                # Extract mass information
                mass = line[130:134].strip()
                # Return the mass as a float if available
                return float(mass) if mass else None

def get_star_properties(hd_number):
    """
    Query the Vizier catalog for a star with the given HD number
    and return its temperature and estimated luminosity.

    Parameters:
    hd_number (str): The HD number of the star to query.

    Returns:
    tuple: A tuple containing the temperature in Kelvin and the estimated luminosity in solar units.
           Returns (None, None) if no data is found.
    """
    # Define the Vizier catalog and set a limit for results
    catalog = "V/117A"
    Vizier.ROW_LIMIT = 1  # Limit to one result to avoid large datasets

    # Query Vizier for the object in the specified catalog
    result = Vizier.query_object(f"HD {hd_number}", catalog=catalog)

    # Check if results were found
    if result:
        # Retrieve the data from the first table in the result
        table = result[0]

        # Extract and calculate temperature
        logTe = table['logTe'][0] if 'logTe' in table.colnames else None
        temperature = 10 ** logTe if logTe else None

        # Estimate luminosity if VMAG is available
        VMAG = table['VMAG'][0] if 'VMAG' in table.colnames else None
        luminosity = 10 ** (0.4 * (4.83 - VMAG)) if VMAG else None

        return temperature, luminosity
    else:
        return None, None

def get_star_properties_with_retries(hd_number, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return get_star_properties(hd_number)
        except ConnectionError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

import re

def get_stellar_type_dr3(gaia_dr3_id, retries=3, delay=5):
    # Customize the Simbad query to include only the spectral type
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('sptype')

    for attempt in range(retries):
        try:
            # Query SIMBAD using Gaia DR3 ID
            result_table = custom_simbad.query_object(f"Gaia DR3 {gaia_dr3_id}")

            if result_table is None:
                print(f"No data found for Gaia DR3 ID {gaia_dr3_id}.")
                return None, None

            # Extract the spectral type
            original_spectral_type = result_table['SP_TYPE'][0] if 'SP_TYPE' in result_table.colnames else None
            processed_spectral_type = None

            if original_spectral_type:
                # Use regex to extract the main spectral type, including formats like K1/2V, K2IV-V, K3+V, or K5/M0V
                # Allow for ".0" or ".5" in the spectral type and handle them appropriately
                match = re.match(r"([A-Z]+[0-9]+(?:/[A-Z]?[0-9]+)?(?:\.0|\.5)?(?:IV-V|IV|V|\+V|\-V)?)", original_spectral_type)
                if match:
                    processed_spectral_type = match.group(1)
                    # Treat K3+V and K3-V as K3V
                    processed_spectral_type = re.sub(r"([A-Z]+[0-9]+)[\+\-]V", r"\1V", processed_spectral_type)
                    # Treat M3.0V as M3V
                    processed_spectral_type = re.sub(r"([A-Z]+[0-9]+)\.0V", r"\1V", processed_spectral_type)

            return original_spectral_type, processed_spectral_type

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None, None

def get_stellar_type_dr2(gaia_dr2_id, retries=3, delay=5):
    # Customize the Simbad query to include only the spectral type
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('sptype')

    for attempt in range(retries):
        try:
            # Query SIMBAD using Gaia DR2 ID
            result_table = custom_simbad.query_object(f"Gaia DR2 {gaia_dr2_id}")

            if result_table is None:
                print(f"No data found for Gaia DR2 ID {gaia_dr2_id}.")
                return None, None

            # Extract the spectral type
            original_spectral_type = result_table['SP_TYPE'][0] if 'SP_TYPE' in result_table.colnames else None
            processed_spectral_type = None

            if original_spectral_type:
                # Use regex to extract the main spectral type, including formats like K1/2V, K2IV-V, K3+V, or K5/M0V
                # Allow for ".0" or ".5" in the spectral type and handle them appropriately
                match = re.match(r"([A-Z]+[0-9]+(?:/[A-Z]?[0-9]+)?(?:\.0|\.5)?(?:IV-V|IV|V|\+V|\-V)?)", original_spectral_type)
                if match:
                    processed_spectral_type = match.group(1)
                    # Treat K3+V and K3-V as K3V
                    processed_spectral_type = re.sub(r"([A-Z]+[0-9]+)[\+\-]V", r"\1V", processed_spectral_type)
                    # Treat M3.0V as M3V
                    processed_spectral_type = re.sub(r"([A-Z]+[0-9]+)\.0V", r"\1V", processed_spectral_type)

            return original_spectral_type, processed_spectral_type

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None, None


def get_stellar_properties_from_gaia(dataframe):
    # Make a copy of the DataFrame to modify
    dataframe_copy = dataframe.copy()

    for index, row in dataframe_copy.iterrows():
        gaia_dr3_id = row['source_id_dr3']
        gaia_dr2_id = row['source_id_dr2']

        # Check for missing properties
        missing_properties = {
            'T_eff [K]': pd.isna(row['T_eff [K]']),
            'Mass [M_Sun]': pd.isna(row['Mass [M_Sun]']),
            'Luminosity [L_Sun]': pd.isna(row['Luminosity [L_Sun]']),
            'Radius [R_Sun]': pd.isna(row['Radius [R_Sun]'])
        }

        if not any(missing_properties.values()):
            continue

        if pd.notna(gaia_dr3_id):
            stellar_type_original, stellar_type = get_stellar_type_dr3(gaia_dr3_id)
        else:
            stellar_type_original, stellar_type = get_stellar_type_dr2(gaia_dr2_id)

        print(index, row['source_id'], stellar_type_original, stellar_type)
        logging.info(f"{index, row['source_id'], stellar_type_original, stellar_type}")

        if stellar_type is None:
            continue

        # Handle spectral types with decimals, slashes, or 'IV-V' by averaging
        if '.5' in stellar_type or '/' in stellar_type or 'IV-V' in stellar_type:
            if '.5' in stellar_type:
                base_numeric = stellar_type.split('.')[0][-1]
                base_letter = stellar_type.split('.')[0][:-1]
                next_numeric = str(int(base_numeric) + 1)

                base_type = base_letter + base_numeric + stellar_type[-1]
                next_type = base_letter + next_numeric + stellar_type[-1]

            elif '/' in stellar_type:
                base_part, next_part = stellar_type.split('/')

                if next_part[0].isalpha():  # K5/M0V
                    base_type = base_part + next_part[2:]
                    next_type = next_part
                elif next_part[0].isdigit():  # K1/2V
                    base_type = base_part + next_part[1:]
                    next_type = base_part[0] + next_part

            elif 'IV-V' in stellar_type:
                base_type = stellar_type.replace('IV-V', 'IV')
                next_type = stellar_type.replace('IV-V', 'V')

            classification_row_base = classification_df[classification_df['Stellar Type'] == base_type]
            classification_row_next = classification_df[classification_df['Stellar Type'] == next_type]

            print(base_type, next_type)
            logging.info(f"Base Type: {base_type}, Next Type: {next_type}")

            if classification_row_base.empty or classification_row_next.empty:
                print(f"-- No data found for stellar type {stellar_type}.")
                logging.warning(f"No data found for stellar type {stellar_type}.")
                continue

            properties = {
                'Mass [M_Sun]': (classification_row_base['Mass Mstar/Msun'].values[0] + classification_row_next['Mass Mstar/Msun'].values[0]) / 2,
                'Luminosity [L_Sun]': (classification_row_base['Luminosity Lstar/Lsun'].values[0] + classification_row_next['Luminosity Lstar/Lsun'].values[0]) / 2,
                'Radius [R_Sun]': (classification_row_base['Radius Rstar/Rsun'].values[0] + classification_row_next['Radius Rstar/Rsun'].values[0]) / 2,
                'T_eff [K]': (classification_row_base['Temp K'].values[0] + classification_row_next['Temp K'].values[0]) / 2
            }
        else:
            classification_row = classification_df[classification_df['Stellar Type'] == stellar_type]

            if classification_row.empty:
                print(f"-- No data found for stellar type (2) {stellar_type}.")
                logging.warning(f"No data found for stellar type (2) {stellar_type}.")
                continue

            properties = {
                'Mass [M_Sun]': classification_row['Mass Mstar/Msun'].values[0],
                'Luminosity [L_Sun]': classification_row['Luminosity Lstar/Lsun'].values[0],
                'Radius [R_Sun]': classification_row['Radius Rstar/Rsun'].values[0],
                'T_eff [K]': classification_row['Temp K'].values[0]
            }

        # Retrieve missing properties from Simbad if necessary
        if any(missing_properties.values()):
            for attempt in range(3):
                try:
                    simbad_result = Simbad.query_object(row['source_id'])
                    if simbad_result is not None:
                        if missing_properties['T_eff [K]']:
                            properties['T_eff [K]'] = simbad_result['Teff'].data[0]
                        if missing_properties['Mass [M_Sun]']:
                            properties['Mass [M_Sun]'] = simbad_result['mass'].data[0]
                        if missing_properties['Luminosity [L_Sun]']:
                            properties['Luminosity [L_Sun]'] = simbad_result['luminosity'].data[0]
                        if missing_properties['Radius [R_Sun]']:
                            properties['Radius [R_Sun]'] = simbad_result['radius'].data[0]
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} to query Simbad failed: {e}")
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"Failed to retrieve data from Simbad for {row['source_id']} after 3 attempts.")
                        logging.error(f"Failed to retrieve data from Simbad for {row['source_id']} after 3 attempts.")
                        continue

        # Update the DataFrame copy with the retrieved properties
        for key, value in properties.items():
            if pd.isna(row[key]):
                dataframe_copy.at[index, key] = value

    return dataframe_copy