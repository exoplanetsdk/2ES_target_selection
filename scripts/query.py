import requests
from astroquery.gaia import Gaia
from formating import *

def execute_gaia_query(query, str_columns=None, output_file=None):
    """
    Execute a Gaia query and optionally save results to Excel
    
    Parameters:
    -----------
    query : str
        The ADQL query to execute
    str_columns : list, optional
        List of column names to convert to string type
    output_file : str, optional
        Path to save Excel file. If None, no file is saved
        
    Returns:
    --------
    pandas.DataFrame
        Query results as a dataframe
    """
    try:
        # Execute query
        job = Gaia.launch_job_async(query)
        df = job.get_results().to_pandas()
        
        # Convert specified columns to string
        if str_columns:
            for col in str_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        
        # Save to Excel if filename provided
        if output_file:
            df.to_excel(output_file, index=False)
            adjust_column_widths(output_file)
            
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"An HTTP error occurred: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None