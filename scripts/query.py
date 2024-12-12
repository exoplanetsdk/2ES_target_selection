import requests
from astroquery.gaia import Gaia
from openpyxl import load_workbook
import pandas as pd
import time
import logging


def adjust_column_widths(excel_file):
    """
    Adjusts the column widths of an Excel workbook based on the maximum length of data in each column,
    and saves the updated workbook.

    Parameters:
    - excel_file: str, the path to the Excel file to be processed.
    """
    # Load the workbook and select the active worksheet
    workbook = load_workbook(excel_file)
    worksheet = workbook.active

    # Adjust the column widths
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter  # Get the column letter
        for cell in column:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except Exception:
                continue
        worksheet.column_dimensions[column_letter].width = max_length + 2

    # Save the workbook
    workbook.save(excel_file)


def execute_gaia_query(query, str_columns=None, output_file=None, retries=3, delay=5):
    """
    Executes a Gaia query and optionally saves the results to an Excel file.

    Parameters:
    -----------
    query : str
        The ADQL query to execute.
    str_columns : list, optional
        List of column names to convert to string type.
    output_file : str, optional
        Path to save the Excel file. If None, no file is saved.
    retries : int
        Number of times to retry the query in case of failure.
    delay : int
        Delay in seconds between retries, with exponential backoff.

    Returns:
    --------
    pandas.DataFrame or None
        Query results as a DataFrame, or None if the query fails.
    """
    
    # Suppress the specific info message from astroquery
    logging.getLogger('astroquery').setLevel(logging.ERROR)

    attempt = 0
    while attempt < retries:
        try:
            # Execute the query
            job = Gaia.launch_job_async(query)
            df = job.get_results().to_pandas()

            # Convert specified columns to string
            if str_columns:
                for col in str_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

            # Save to Excel if a filename is provided
            if output_file:
                df.to_excel(output_file, index=False)
                adjust_column_widths(output_file)

            # print(f"Number of results: {len(df)}")
            return df

        except requests.exceptions.HTTPError as e:
            print(f"An HTTP error occurred: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"A connection error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        attempt += 1
        print(f"Retrying in {delay} seconds... (Attempt {attempt}/{retries})")
        time.sleep(delay)
        delay *= 2  # Exponential backoff

    print("Failed to execute query after several attempts.")
    return None


