import time
import re
import logging
import pandas as pd
import warnings
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.units import UnitsWarning
from requests.exceptions import ConnectionError

# Configure logging
# logging.basicConfig(
#     filename='stellar_properties_log.txt',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# Suppress specific warning
warnings.simplefilter('ignore', category=UnitsWarning)

# Load classification data
classification_df = pd.read_csv(
    "../data/EEM_dwarf_UBVIJHK_colors_Teff.txt",
    delim_whitespace=True,
    skiprows=22,
    nrows=118,
    header=0,
    na_values='...'
)

def get_simbad_info_with_retry(source_id, retries=3, delay=5):
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('ids', 'otype')
    
    for attempt in range(retries):
        try:
            result_table = custom_simbad.query_object(f"Gaia DR3 {source_id}")
            if result_table is not None:
                ids = result_table['IDS'][0].split('|')
                return {
                    'HD Number': ', '.join([id.strip() for id in ids if id.startswith('HD')]) or None,
                    'GJ Number': ', '.join([id.strip() for id in ids if id.startswith('GJ')]) or None,
                    'HIP Number': ', '.join([id.strip() for id in ids if id.startswith('HIP')]) or None,
                    'Object Type': result_table['OTYPE'][0]
                }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None

def get_star_properties(hd_number):
    """Query Vizier catalog for star properties."""
    catalog = "V/117A"
    Vizier.ROW_LIMIT = 1

    result = Vizier.query_object(f"HD {hd_number}", catalog=catalog)

    if result:
        table = result[0]
        logTe = table['logTe'][0] if 'logTe' in table.colnames else None
        temperature = 10 ** logTe if logTe else None
        
        VMAG = table['VMAG'][0] if 'VMAG' in table.colnames else None
        luminosity = 10 ** (0.4 * (4.83 - VMAG)) if VMAG else None

        return temperature, luminosity
    return None, None

def get_star_properties_with_retries(hd_number, retries=3, delay=5):
    """Retry wrapper for get_star_properties."""
    for attempt in range(retries):
        try:
            return get_star_properties(hd_number)
        except ConnectionError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

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

def get_empirical_stellar_parameters(dataframe):
    print("Getting empirical stellar parameters")
    # Make a copy of the DataFrame to modify
    dataframe_copy = dataframe.copy()

    for column in ['Stellar Parameter Source', 'SIMBAD Spectral Type', 'Readable Spectral Type (experimental)']:
        if column not in dataframe_copy.columns:
            dataframe_copy[column] = None

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

        if pd.notna(gaia_dr3_id):
            stellar_type_original, stellar_type = get_stellar_type_dr3(gaia_dr3_id)
        else:
            stellar_type_original, stellar_type = get_stellar_type_dr2(gaia_dr2_id)

        dataframe_copy.at[index, 'SIMBAD Spectral Type'] = stellar_type_original
        dataframe_copy.at[index, 'Readable Spectral Type (experimental)'] = stellar_type

        print(index, row['source_id'], stellar_type_original, stellar_type)
        # logging.info(f"{index, row['source_id'], stellar_type_original, stellar_type}")

        if stellar_type is None:
            continue

        if not any(missing_properties.values()):
            dataframe_copy.at[index, 'Stellar Parameter Source'] = 'GAIA'
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
                base_type = stellar_type.replace('IV-V', 'V')
                # base_type = stellar_type.replace('IV-V', 'IV')
                next_type = stellar_type.replace('IV-V', 'V')

            classification_row_base = classification_df[classification_df['#SpT'] == base_type]
            classification_row_next = classification_df[classification_df['#SpT'] == next_type]

            dataframe_copy.at[index, 'Readable Spectral Type (experimental)'] = str(base_type) + ' and ' + str(next_type)
            print(base_type, next_type)
            # logging.info(f"Base Type: {base_type}, Next Type: {next_type}")

            if classification_row_base.empty or classification_row_next.empty:
                print(f"-- No data found for stellar type {stellar_type}.")
                # logging.warning(f"No data found for stellar type {stellar_type}.")
                continue

            properties = {
                'Mass [M_Sun]': (classification_row_base['Msun'].values[0] + classification_row_next['Msun'].values[0]) / 2,
                'Luminosity [L_Sun]': (10**classification_row_base['logL'].values[0] + 10**classification_row_next['logL'].values[0]) / 2,
                'Radius [R_Sun]': (classification_row_base['R_Rsun'].values[0] + classification_row_next['R_Rsun'].values[0]) / 2,
                'T_eff [K]': (classification_row_base['Teff'].values[0] + classification_row_next['Teff'].values[0]) / 2,
            }
        else:
            classification_row = classification_df[classification_df['#SpT'] == stellar_type]

            if classification_row.empty:
                print(f"-- No data found for stellar type (2) {stellar_type}.")
                # logging.warning(f"No data found for stellar type (2) {stellar_type}.")
                continue

            properties = {
                'Mass [M_Sun]': classification_row['Msun'].values[0],
                'Luminosity [L_Sun]': 10**classification_row['logL'].values[0],
                'Radius [R_Sun]': classification_row['R_Rsun'].values[0],
                'T_eff [K]': classification_row['Teff'].values[0],
            }

        # Retrieve missing properties from Simbad if necessary
        if any(missing_properties.values()):
            dataframe_copy.at[index, 'Stellar Parameter Source'] = 'SIMBAD + empirical'
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
                        # logging.error(f"Failed to retrieve data from Simbad for {row['source_id']} after 3 attempts.")
                        continue

        # Update the DataFrame copy with the retrieved properties
        for key, value in properties.items():
            if pd.isna(row[key]):
                dataframe_copy.at[index, key] = value

    return dataframe_copy