    AND NOT EXISTS (
        SELECT 1
        FROM gaiadr2.gaia_source AS neighbors
        WHERE 
            1 = CONTAINS(
                POINT('ICRS', gs.ra, gs.dec),
                CIRCLE('ICRS', neighbors.ra, neighbors.dec, 2/3600.0)
            )
            AND neighbors.phot_g_mean_mag < {NEIGHBOR_G_MAG_LIMIT}
            AND gs.source_id != neighbors.source_id
    )


#----------------------------------------------------------------



    AND NOT EXISTS (
        SELECT 1
        FROM gaiadr3.gaia_source AS neighbors
        WHERE 
            1 = CONTAINS(
                POINT('ICRS', gs.ra, gs.dec),
                CIRCLE('ICRS', neighbors.ra, neighbors.dec, 2/3600.0)
            )
            AND neighbors.phot_g_mean_mag < {NEIGHBOR_G_MAG_LIMIT}
            AND gs.source_id != neighbors.source_id
    )

#----------------------------------------------------------------
# Batch query for Gaia DR2 in parallel
#----------------------------------------------------------------

import pandas as pd
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_gaia_query_segment(query, str_columns=None):
    """
    Execute a Gaia query segment and return results as a DataFrame.
    
    Parameters:
    -----------
    query : str
        The ADQL query segment to execute
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Query results as a dataframe, or an empty dataframe if the query fails
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
        
        print(f"Number of results: {len(df)}")
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return pd.DataFrame()  # Return an empty DataFrame if the query fails

def execute_gaia_query_in_batches(query_template, ra_ranges, str_columns=None):
    """
    Execute Gaia queries in batches based on RA ranges and combine results.
    
    Parameters:
    -----------
    query_template : str
        Template for the ADQL query with placeholders for RA range.
    ra_ranges : list of tuples
        List of (min_ra, max_ra) tuples to query in batches.
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Combined query results as a dataframe
    """
    all_results = []
    
    def query_ra_range(min_ra, max_ra):
        # Format the query with the current RA range
        query = query_template.format(min_ra=min_ra, max_ra=max_ra)
        # Execute the query segment and return results
        return execute_gaia_query_segment(query, str_columns=str_columns)
    
    # Use ThreadPoolExecutor to run queries in parallel
    with ThreadPoolExecutor() as executor:
        future_to_ra = {executor.submit(query_ra_range, min_ra, max_ra): (min_ra, max_ra) for min_ra, max_ra in ra_ranges}
        for future in as_completed(future_to_ra):
            ra_range = future_to_ra[future]
            try:
                df = future.result()
                if not df.empty:
                    all_results.append(df)
            except Exception as e:
                print(f"An error occurred for RA range {ra_range}: {e}")
    
    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

# Define RA ranges for batch processing
ra_ranges = [(i, i + 1) for i in range(0, 360)]

# Adjust query template to include RA range
query_template = query_dr2.replace("WHERE", "WHERE gs.ra BETWEEN {min_ra} AND {max_ra} AND")

# Execute queries in batches and combine results
df_dr2_combined = execute_gaia_query_in_batches(
    query_template,
    ra_ranges,
    str_columns=['source_id']
)

# Save combined results to Excel
if not df_dr2_combined.empty:
    df_dr2_combined.to_excel(directory + 'dr2_results_combined.xlsx', index=False)
    adjust_column_widths(directory + 'dr2_results_combined.xlsx')


#----------------------------------------------------------------
# Batch query for Gaia DR2 in truncated serial
#----------------------------------------------------------------   

# Note: it's taking too long to run (over 24 hours and still running)

import pandas as pd
import time

def execute_gaia_query_segment(query, str_columns=None):
    """
    Execute a Gaia query segment and return results as a DataFrame.
    
    Parameters:
    -----------
    query : str
        The ADQL query segment to execute
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Query results as a dataframe, or an empty dataframe if the query fails
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
        
        print(f"Number of results: {len(df)}")
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return pd.DataFrame()  # Return an empty DataFrame if the query fails

def execute_gaia_query_in_batches(query_template, ra_ranges, str_columns=None):
    """
    Execute Gaia queries in batches based on RA ranges and combine results.
    
    Parameters:
    -----------
    query_template : str
        Template for the ADQL query with placeholders for RA range.
    ra_ranges : list of tuples
        List of (min_ra, max_ra) tuples to query in batches.
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Combined query results as a dataframe
    """
    all_results = []
    
    for min_ra, max_ra in ra_ranges:
        # Format the query with the current RA range
        query = query_template.format(min_ra=min_ra, max_ra=max_ra)
        
        # Execute the query segment and collect results
        df = execute_gaia_query_segment(query, str_columns=str_columns)
        
        if not df.empty:
            all_results.append(df)
    
    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

# Define RA ranges for batch processing
ra_ranges = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]

# Adjust query template to include RA range
query_template = query_dr2.replace("WHERE", "WHERE gs.ra BETWEEN {min_ra} AND {max_ra} AND")

# Execute queries in batches and combine results
df_dr2_combined = execute_gaia_query_in_batches(
    query_template,
    ra_ranges,
    str_columns=['source_id']
)

# Save combined results to Excel
if not df_dr2_combined.empty:
    df_dr2_combined.to_excel(directory + 'dr2_results_combined.xlsx', index=False)
    adjust_column_widths(directory + 'dr2_results_combined.xlsx')
