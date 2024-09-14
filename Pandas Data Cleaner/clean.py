import re
import pandas as pd
import numpy as np
from time import perf_counter
import time

# Constants
EMPTY_THRESHOLD = 0.5
LOW_COUNT_THRESHOLD = 2
VALID_DATA_THRESHOLD = 0.5

def print_dataframe_info(df, step=""):
    num_columns = len(df.columns)
    num_rows = len(df)
    num_cells = num_columns * num_rows
    print(f"{step}Dataframe info:")
    print(f"  Number of columns: {num_columns}")
    print(f"  Number of rows: {num_rows}")
    print(f"  Total number of cells: {num_cells}")

def check_and_normalize_column_headers(df):
    print("Checking and normalizing column headers...")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.columns = [re.sub(r'[^0-9a-zA-Z_]', '', col) for col in df.columns]
    print("Column names have been normalized.")
    return df

def remove_empty_columns(df, threshold=EMPTY_THRESHOLD):
    print(f"Removing columns with less than {threshold * 100}% valid data...")
    return df.dropna(axis=1, thresh=int(threshold * len(df)))

def remove_empty_rows(df, threshold=EMPTY_THRESHOLD):
    print(f"Removing rows with less than {threshold * 100}% valid data...")
    return df.dropna(thresh=int(threshold * len(df.columns)))

def drop_rows_with_nas(df, threshold=VALID_DATA_THRESHOLD):
    print(f"Dropping rows with NAs for columns with more than {threshold * 100}% valid data...")
    valid_columns = df.columns[df.notna().mean() > threshold]
    return df.dropna(subset=valid_columns)

def check_typos(df, column_name, threshold=2, top_n=100):
    if df[column_name].dtype != 'object':
        print(f"Skipping typo check for column {column_name} as it is not a string type.")
        return None

    print(f"Checking for typos in column: {column_name}")

    try:
        value_counts = df[column_name].value_counts()
        top_values = value_counts.head(top_n).index.tolist()

        def find_similar_strings(value):
            if pd.isna(value):
                return []
            return [tv for tv in top_values if value != tv and levenshtein_distance(value, tv) <= threshold]

        df['possible_typos'] = df[column_name].apply(find_similar_strings)
        typos_df = df[df['possible_typos'].apply(len) > 0][[column_name, 'possible_typos']]

        typo_count = len(typos_df)
        if typo_count > 0:
            print(f"Potential typos found in column {column_name}: {typo_count}")
            print(typos_df.head(10))
            return typos_df
        else:
            print(f"No potential typos found in column {column_name}")
            return None

    except Exception as e:
        print(f"Unexpected error in check_typos for column {column_name}: {str(e)}")
        return None

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def transform_string_column(df, column_name):
    print(f"Transforming string column: {column_name}")
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True)
    df[column_name] = df[column_name].str.replace(r'[^a-zA-Z0-9\s/:.-]', '', regex=True)
    return df

def clean_column(df, column_name):
    print(f"Cleaning column: {column_name}")
    start_time = perf_counter()

    if df[column_name].dtype == 'object':
        typos_df = check_typos(df, column_name)
        if typos_df is not None and len(typos_df) > 0:
            print(f"Detailed typos for column {column_name}:")
            print(typos_df)
        df = transform_string_column(df, column_name)
    elif pd.api.types.is_numeric_dtype(df[column_name]):
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    end_time = perf_counter()
    print(f"Time taken to clean {column_name}: {end_time - start_time:.6f} seconds")
    return df

def remove_outliers(df, column):
    print(f"Removing outliers from column: {column}")
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def calculate_nonconforming_cells(df):
    return df.isna().sum().to_dict()

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def remove_duplicates_from_primary_key(df, primary_key_column):
    print(f"Removing duplicates based on primary key column: {primary_key_column}")
    return df.drop_duplicates(subset=[primary_key_column])

def clean_data(df, primary_key_column, progress):
    start_time = time.time()
    process_times = {}

    print("Starting data validation and cleaning...")
    print_dataframe_info(df, "Initial - ")

    nonconforming_cells_before = calculate_nonconforming_cells(df)

    progress(0.1, desc="Normalizing column headers")
    step_start_time = time.time()
    df = check_and_normalize_column_headers(df)
    process_times['Normalize headers'] = time.time() - step_start_time

    progress(0.2, desc="Removing empty columns")
    step_start_time = time.time()
    df = remove_empty_columns(df)
    print('2) count of valid rows:', len(df))
    process_times['Remove empty columns'] = time.time() - step_start_time

    progress(0.3, desc="Removing empty rows")
    step_start_time = time.time()
    df = remove_empty_rows(df)
    print('3) count of valid rows:', len(df))
    process_times['Remove empty rows'] = time.time() - step_start_time

    progress(0.4, desc="Dropping rows with NAs")
    step_start_time = time.time()
    df = drop_rows_with_nas(df)
    print('4) count of valid rows:', len(df))
    process_times['Drop rows with NAs'] = time.time() - step_start_time

    column_cleaning_times = {}
    total_columns = len(df.columns)
    for index, column in enumerate(df.columns):
        progress(0.5 + (0.2 * (index / total_columns)), desc=f"Cleaning column: {column}")
        column_start_time = time.time()
        df = clean_column(df, column)
        print('5) count of valid rows:', len(df))
        column_cleaning_times[f"Clean column: {column}"] = time.time() - column_start_time
    process_times.update(column_cleaning_times)

    progress(0.7, desc="Removing outliers")
    step_start_time = time.time()
    numeric_columns = get_numeric_columns(df)
    numeric_columns = [col for col in numeric_columns if col != primary_key_column]
    for column in numeric_columns:
        df = remove_outliers(df, column)
    print('6) count of valid rows:', len(df))
    process_times['Remove outliers'] = time.time() - step_start_time

    progress(0.8, desc="Removing duplicates from primary key")
    step_start_time = time.time()
    df = remove_duplicates_from_primary_key(df, primary_key_column)
    print('7) count of valid rows:', len(df))

    print("Cleaning process completed.")
    print_dataframe_info(df, "Final - ")

    return df, nonconforming_cells_before, process_times