import pandas as pd
from src.fraud_detection.config.configuration import ConfigurationManager
import chardet
from pathlib import Path
from src.fraud_detection import logger
import csv


async def import_data(uploaded_file):
    # Create configuration manager and get data ingestion config
    config_manager = ConfigurationManager()
    config = config_manager.get_data_ingestion_config()
        
    path_save = Path(config.local_data_file)  
    df = None

    if uploaded_file is not None:
        path_save.parent.mkdir(parents=True, exist_ok=True)

        filename = uploaded_file.filename
        if not filename:
            raise ValueError("Uploaded file has no filename.")

        # Save the uploaded file
        with open(path_save, "wb") as f:  
            content = await uploaded_file.read()
            f.write(content)

        try:
            sample = content.decode('utf-8', errors='replace')[:4096]
            encoding = 'utf-8'
        except Exception:
            result = chardet.detect(content)
            encoding = result['encoding'] or 'utf-8'
            sample = content.decode(encoding, errors='replace')[:4096]

        # Read CSV or Excel
        if filename.endswith(('.csv', '.txt')):
            delimiter = ','
            for delim in [',', ';', '\t', '|']:
                if sample.count(delim) > 1:
                    delimiter = delim
                    break
            try:
                df = pd.read_csv(path_save, encoding=encoding, delimiter=delimiter, 
                                low_memory=False, on_bad_lines='skip')
            except pd.errors.EmptyDataError:
                raise ValueError("No data found in the CSV file.")
            except pd.errors.ParserError as e:
                raise ValueError(f"Error reading CSV: {str(e)}")
            except Exception as e:  # Broader exception for unexpected issues
                raise ValueError(f"An unexpected error occurred while reading CSV: {str(e)}")

        elif filename.lower().endswith(('.xlsx', '.xls')):  # Case-insensitive check for Excel
            engine = 'openpyxl' if filename.lower().endswith('.xlsx') else 'xlrd'
            try:
                df = pd.read_excel(path_save, engine=engine)
            except Exception as e:  # Handle Excel reading errors
                raise ValueError(f"Error reading Excel file: {str(e)}")

        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel files.")

        if df is None:
            raise ValueError("Pandas DataFrame was not initialized. Check file format or reading process.")
        if df.empty:
            raise ValueError("No data found in the file. Please check the file content.")

        return df

    else:
        raise ValueError("No file provided.")  # Consistent error message
