import requests
from astroquery.gaia import Gaia
from openpyxl import load_workbook


def adjust_column_widths(excel_file):
    """
    Load an Excel workbook, adjust the column widths based on the maximum length of data in each column,
    and save the workbook.

    Parameters:
    - directory: str, the path to the directory containing the Excel file.
    - excel_file: str, the name of the Excel file to be processed.
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
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column_letter].width = adjusted_width

    # Save the workbook
    workbook.save(excel_file)


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
            
        print(f"Number of results: {len(df)}")
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"An HTTP error occurred: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

