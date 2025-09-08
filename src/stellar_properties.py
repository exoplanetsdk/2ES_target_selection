import time
import re
import logging
import pandas as pd
import warnings
import os
import pickle
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.units import UnitsWarning
from requests.exceptions import ConnectionError
from tqdm import tqdm
from config import *

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
    sep=r'\s+',
    skiprows=22,
    nrows=118,
    header=0,
    na_values='...'
)

#------------------------------------------------------------------------------------------------

# Update 2025-07-22: 
# - specify the Gaia data release (e.g., Gaia DR3) to ensure identifiers are extracted in SIMBAD
# e.g., gaia_identifier = 'Gaia DR3 2452378776434477184'

def get_simbad_info_with_retry(gaia_identifier, max_retries=3, delay=1):
    """
    Query SIMBAD with Gaia identifier and extract specific catalog numbers
    
    Parameters:
    -----------
    gaia_identifier : str
        Full Gaia identifier (e.g., 'Gaia DR3 2452378776434477184')
    max_retries : int
        Maximum number of retry attempts
    delay : float
        Delay between retries in seconds
    
    Returns:
    --------
    dict or None
        Dictionary containing HD Number, GJ Number, HIP Number, and Object Type
        Returns None if query fails or object not found
    """
    
    # Configure SIMBAD to return identifiers and object type
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields('ids', 'otype', 'main_id')
    
    for attempt in range(max_retries):
        try:
            # Query SIMBAD with the Gaia identifier
            result = Simbad.query_object(gaia_identifier)
            
            if result is None or len(result) == 0:
                return None
            
            # Extract the identifiers and object type
            ids_string = ""
            object_type = ""
            main_id = ""
            
            # Handle different possible column names
            if 'IDS' in result.colnames:
                ids_string = str(result['IDS'][0])
            elif 'ids' in result.colnames:
                ids_string = str(result['ids'][0])
                
            if 'OTYPE' in result.colnames:
                object_type = str(result['OTYPE'][0])
            elif 'otype' in result.colnames:
                object_type = str(result['otype'][0])
                
            if 'MAIN_ID' in result.colnames:
                main_id = str(result['MAIN_ID'][0])
            elif 'main_id' in result.colnames:
                main_id = str(result['main_id'][0])
            
            # Initialize result dictionary
            simbad_info = {
                'HD Number': None,
                'GJ Number': None, 
                'HIP Number': None,
                'Object Type': object_type.strip() if object_type else None
            }
            
            # Parse identifiers from both ids_string and main_id
            all_identifiers = []
            
            if ids_string and ids_string != 'nan':
                # Split by common delimiters
                for delimiter in ['|', '\n', ';']:
                    if delimiter in ids_string:
                        all_identifiers.extend([id_str.strip() for id_str in ids_string.split(delimiter)])
                        break
                else:
                    # No delimiter found, treat as single identifier
                    all_identifiers.append(ids_string.strip())
            
            if main_id and main_id != 'nan':
                all_identifiers.append(main_id.strip())
            
            # Extract catalog numbers from all identifiers
            for identifier in all_identifiers:
                if not identifier or identifier == 'nan':
                    continue
                    
                # Extract HD number (various formats)
                hd_patterns = [
                    r'HD\s*(\d+)',
                    r'HD(\d+)',
                    r'Henry\s*Draper\s*(\d+)'
                ]
                for pattern in hd_patterns:
                    hd_match = re.search(pattern, identifier, re.IGNORECASE)
                    if hd_match and simbad_info['HD Number'] is None:
                        simbad_info['HD Number'] = f"HD {hd_match.group(1)}"
                        break
                
                # Extract GJ number (Gliese-Jahreiss catalog)
                gj_patterns = [
                    r'GJ\s*(\d+(?:\.\d+)?[A-Z]*)',
                    r'Gl\s*(\d+(?:\.\d+)?[A-Z]*)',
                    r'Gliese\s*(\d+(?:\.\d+)?[A-Z]*)'
                ]
                for pattern in gj_patterns:
                    gj_match = re.search(pattern, identifier, re.IGNORECASE)
                    if gj_match and simbad_info['GJ Number'] is None:
                        simbad_info['GJ Number'] = f"GJ {gj_match.group(1)}"
                        break
                
                # Extract HIP number (Hipparcos catalog)
                hip_patterns = [
                    r'HIP\s*(\d+)',
                    r'Hipparcos\s*(\d+)'
                ]
                for pattern in hip_patterns:
                    hip_match = re.search(pattern, identifier, re.IGNORECASE)
                    if hip_match and simbad_info['HIP Number'] is None:
                        simbad_info['HIP Number'] = f"HIP {hip_match.group(1)}"
                        break
            
            return simbad_info
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {gaia_identifier}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed for {gaia_identifier}")
                return None
    
    return None

#------------------------------------------------------------------------------------------------

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

def get_stellar_type(gaia_id, retries=3, delay=5):
    # Customize the Simbad query to include only the spectral type
    try:
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields('sp_type')
    except Exception as e:
        logging.error(f"SIMBAD configuration failed: {e}")
        return "SERVICE_DOWN", "SERVICE_DOWN"

    for attempt in range(retries):
        try:
            # Query SIMBAD using Gaia ID
            result_table = Simbad.query_object(gaia_id)

            if result_table is None:
                logging.warning(f"No data found for {gaia_id}.")
                return None, None

            # Extract the spectral type
            original_spectral_type = result_table['sp_type'][0] if 'sp_type' in result_table.colnames else None
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

#------------------------------------------------------------------------------------------------

def get_empirical_stellar_parameters(dataframe, stellar_type_cache_path=RESULTS_DIRECTORY+"stellar_type_cache.pkl"):
    print("\nGetting empirical stellar parameters")

    # Configure logging to use tqdm.write
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    # Set up logging with the custom handler
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(TqdmLoggingHandler())

    # Make a copy of the DataFrame to modify
    dataframe_copy = dataframe.copy()

    for column in ['Stellar Parameter Source', 'SIMBAD Spectral Type', 'Readable Spectral Type (experimental)']:
        if column not in dataframe_copy.columns:
            dataframe_copy[column] = None

    # --- Load or initialize the stellar type cache ---
    if os.path.exists(stellar_type_cache_path):
        try:
            with open(stellar_type_cache_path, "rb") as f:
                stellar_type_cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load stellar type cache: {e}")
            stellar_type_cache = {}
    else:
        stellar_type_cache = {}

    updated_cache = False

    for index, row in tqdm(dataframe_copy.iterrows(), 
                          total=dataframe_copy.shape[0], 
                          desc="Processing stellar types",
                          ncols=100):  # Fixed width progress bar

        # Check for missing properties
        missing_properties = {
            'T_eff [K]': pd.isna(row['T_eff [K]']),
            'Mass [M_Sun]': pd.isna(row['Mass [M_Sun]']),
            'Luminosity [L_Sun]': pd.isna(row['Luminosity [L_Sun]']),
            'Radius [R_Sun]': pd.isna(row['Radius [R_Sun]'])
        }

        gaia_id = row['source_id']

        # --- Use cache if available, otherwise query and update cache ---
        if gaia_id in stellar_type_cache:
            stellar_type_original, stellar_type = stellar_type_cache[gaia_id]
        else:
            stellar_type_original, stellar_type = get_stellar_type(gaia_id)
            stellar_type_cache[gaia_id] = (stellar_type_original, stellar_type)
            updated_cache = True

        dataframe_copy.at[index, 'SIMBAD Spectral Type'] = stellar_type_original
        dataframe_copy.at[index, 'Readable Spectral Type (experimental)'] = stellar_type

        # print(index, row['source_id'], stellar_type_original, stellar_type, end='\r')
        logging.warning(f"{index}: {row['source_id']}, {stellar_type_original} --> {stellar_type}")

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
            # print(base_type, next_type)
            logging.warning(f"Lower Type: {base_type}, Higher Type: {next_type}")

            if classification_row_base.empty or classification_row_next.empty:
                # print(f"-- No data found for stellar type {stellar_type}.\r")
                logging.warning(f"No data found for stellar type {stellar_type}.")
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
                # print(f"-- No data found for stellar type (2) {stellar_type}.\r")
                logging.warning(f"No data found for stellar type (2) {stellar_type}.")
                continue

            properties = {
                'Mass [M_Sun]': classification_row['Msun'].values[0],
                'Luminosity [L_Sun]': 10**classification_row['logL'].values[0],
                'Radius [R_Sun]': classification_row['R_Rsun'].values[0],
                'T_eff [K]': classification_row['Teff'].values[0],
            }

        # Update the DataFrame copy with the retrieved properties
        for key, value in properties.items():
            if pd.isna(row[key]):
                dataframe_copy.at[index, key] = value

    # --- Save the updated cache if it was changed ---
    if updated_cache:
        try:
            with open(stellar_type_cache_path, "wb") as f:
                pickle.dump(stellar_type_cache, f)
        except Exception as e:
            print(f"Warning: Could not save stellar type cache: {e}")

    return dataframe_copy