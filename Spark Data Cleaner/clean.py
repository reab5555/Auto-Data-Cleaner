import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, lower, regexp_replace, to_date, to_timestamp, udf, \
    levenshtein, array, lit, trim, size, coalesce
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType, TimestampType, ArrayType
from pyspark.sql.utils import AnalysisException
import time
from time import perf_counter

# Constants
EMPTY_THRESHOLD = 0.5
LOW_COUNT_THRESHOLD = 2
VALID_DATA_THRESHOLD = 0.5

def print_dataframe_info(df, step=""):
    num_columns = len(df.columns)
    num_rows = df.count()
    num_cells = num_columns * num_rows
    print(f"{step}Dataframe info:")
    print(f"  Number of columns: {num_columns}")
    print(f"  Number of rows: {num_rows}")
    print(f"  Total number of cells: {num_cells}")


def check_and_normalize_column_headers(df):
    print("Checking and normalizing column headers...")

    for old_name in df.columns:
        # Create the new name using string manipulation
        new_name = old_name.lower().replace(' ', '_')

        # Remove any non-alphanumeric characters (excluding underscores)
        new_name = re.sub(r'[^0-9a-zA-Z_]', '', new_name)

        # Rename the column
        df = df.withColumnRenamed(old_name, new_name)

    print("Column names have been normalized.")
    return df


def remove_empty_columns(df, threshold=EMPTY_THRESHOLD):
    print(f"Removing columns with less than {threshold * 100}% valid data...")

    # Calculate the percentage of non-null values for each column
    df_stats = df.select(
        [((count(when(col(c).isNotNull(), c)) / count('*')) >= threshold).alias(c) for c in df.columns])
    valid_columns = [c for c in df_stats.columns if df_stats.select(c).first()[0]]

    return df.select(valid_columns)


def remove_empty_rows(df, threshold=EMPTY_THRESHOLD):
    print(f"Removing rows with less than {threshold * 100}% valid data...")

    # Count the number of non-null values for each row
    expr = sum([when(col(c).isNotNull(), lit(1)).otherwise(lit(0)) for c in df.columns])
    df_valid_count = df.withColumn('valid_count', expr)

    # Filter rows based on the threshold
    total_columns = len(df.columns)
    df_filtered = df_valid_count.filter(col('valid_count') >= threshold * total_columns)

    print('count of valid rows:', df_filtered.count())

    return df_filtered.drop('valid_count')


def drop_rows_with_nas(df, threshold=VALID_DATA_THRESHOLD):
    print(f"Dropping rows with NAs for columns with more than {threshold * 100}% valid data...")

    # Calculate the percentage of non-null values for each column
    df_stats = df.select([((count(when(col(c).isNotNull(), c)) / count('*'))).alias(c) for c in df.columns])

    # Get columns with more than threshold valid data
    valid_columns = [c for c in df_stats.columns if df_stats.select(c).first()[0] > threshold]

    # Drop rows with NAs only for the valid columns
    for column in valid_columns:
        df = df.filter(col(column).isNotNull())

    return df

def check_typos(df, column_name, threshold=2, top_n=100):
    # Check if the column is of StringType
    if not isinstance(df.schema[column_name].dataType, StringType):
        print(f"Skipping typo check for column {column_name} as it is not a string type.")
        return None

    print(f"Checking for typos in column: {column_name}")

    try:
        # Get value counts for the specific column
        value_counts = df.groupBy(column_name).count().orderBy("count", ascending=False)

        # Take top N most frequent values
        top_values = [row[column_name] for row in value_counts.limit(top_n).collect()]

        # Broadcast the top values to all nodes
        broadcast_top_values = df.sparkSession.sparkContext.broadcast(top_values)

        # Define UDF to find similar strings
        @udf(returnType=ArrayType(StringType()))
        def find_similar_strings(value):
            if value is None:
                return []
            similar = []
            for top_value in broadcast_top_values.value:
                if value != top_value and levenshtein(value, top_value) <= threshold:
                    similar.append(top_value)
            return similar

        # Apply the UDF to the column
        df_with_typos = df.withColumn("possible_typos", find_similar_strings(col(column_name)))

        # Filter rows with possible typos and select only the relevant columns
        typos_df = df_with_typos.filter(size("possible_typos") > 0).select(column_name, "possible_typos")

        # Check if there are any potential typos
        typo_count = typos_df.count()
        if typo_count > 0:
            print(f"Potential typos found in column {column_name}: {typo_count}")
            typos_df.show(10, truncate=False)
            return typos_df
        else:
            print(f"No potential typos found in column {column_name}")
            return None

    except AnalysisException as e:
        print(f"Error analyzing column {column_name}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error in check_typos for column {column_name}: {str(e)}")
        return None


def transform_string_column(df, column_name):
    print(f"Transforming string column: {column_name}")
    # Lower case transformation (if applicable)
    df = df.withColumn(column_name, lower(col(column_name)))
    # Remove leading and trailing spaces
    df = df.withColumn(column_name, trim(col(column_name)))
    # Replace multiple spaces with a single space
    df = df.withColumn(column_name, regexp_replace(col(column_name), "\\s+", " "))
    # Remove special characters except those used in dates and times
    df = df.withColumn(column_name, regexp_replace(col(column_name), "[^a-zA-Z0-9\\s/:.-]", ""))
    return df


def clean_column(df, column_name):
    print(f"Cleaning column: {column_name}")
    start_time = perf_counter()
    # Get the data type of the current column
    column_type = df.schema[column_name].dataType

    if isinstance(column_type, StringType):
        typos_df = check_typos(df, column_name)
        if typos_df is not None and typos_df.count() > 0:
            print(f"Detailed typos for column {column_name}:")
            typos_df.show(truncate=False)
        df = transform_string_column(df, column_name)

    elif isinstance(column_type, (DoubleType, IntegerType)):
        # For numeric columns, we'll do a simple null check
        df = df.withColumn(column_name, when(col(column_name).isNull(), lit(None)).otherwise(col(column_name)))

    end_time = perf_counter()
    print(f"Time taken to clean {column_name}: {end_time - start_time:.6f} seconds")
    return df

# Update the remove_outliers function to work on a single column
def remove_outliers(df, column):
    print(f"Removing outliers from column: {column}")

    stats = df.select(column).summary("25%", "75%").collect()
    q1 = float(stats[0][1])
    q3 = float(stats[1][1])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))

    return df

def calculate_nonconforming_cells(df):
    nonconforming_cells = {}
    for column in df.columns:
        nonconforming_count = df.filter(col(column).isNull() | isnan(column)).count()
        nonconforming_cells[column] = nonconforming_count
    return nonconforming_cells

def get_numeric_columns(df):
    return [field.name for field in df.schema.fields if isinstance(field.dataType, (IntegerType, DoubleType))]

def remove_duplicates_from_primary_key(df, primary_key_column):
    print(f"Removing duplicates based on primary key column: {primary_key_column}")
    return df.dropDuplicates([primary_key_column])

def clean_data(spark, df, primary_key_column, progress):
    start_time = time.time()
    process_times = {}

    print("Starting data validation and cleaning...")
    print_dataframe_info(df, "Initial - ")

    # Calculate nonconforming cells before cleaning
    nonconforming_cells_before = calculate_nonconforming_cells(df)

    # Step 1: Normalize column headers
    progress(0.1, desc="Normalizing column headers")
    step_start_time = time.time()
    df = check_and_normalize_column_headers(df)
    process_times['Normalize headers'] = time.time() - step_start_time

    # Step 2: Remove empty columns
    progress(0.2, desc="Removing empty columns")
    step_start_time = time.time()
    df = remove_empty_columns(df)
    print('2) count of valid rows:', df.count())
    process_times['Remove empty columns'] = time.time() - step_start_time

    # Step 3: Remove empty rows
    progress(0.3, desc="Removing empty rows")
    step_start_time = time.time()
    df = remove_empty_rows(df)
    print('3) count of valid rows:', df.count())
    process_times['Remove empty rows'] = time.time() - step_start_time

    # Step 4: Drop rows with NAs for columns with more than 50% valid data
    progress(0.4, desc="Dropping rows with NAs")
    step_start_time = time.time()
    df = drop_rows_with_nas(df)
    print('4) count of valid rows:', df.count())
    process_times['Drop rows with NAs'] = time.time() - step_start_time

    # Step 5: Clean columns (including typo checking and string transformation)
    column_cleaning_times = {}
    total_columns = len(df.columns)
    for index, column in enumerate(df.columns):
        progress(0.5 + (0.2 * (index / total_columns)), desc=f"Cleaning column: {column}")
        column_start_time = time.time()
        df = clean_column(df, column)
        print('5) count of valid rows:', df.count())
        column_cleaning_times[f"Clean column: {column}"] = time.time() - column_start_time
    process_times.update(column_cleaning_times)

    # Step 6: Remove outliers from numeric columns (excluding primary key)
    progress(0.7, desc="Removing outliers")
    step_start_time = time.time()
    numeric_columns = get_numeric_columns(df)
    numeric_columns = [col for col in numeric_columns if col != primary_key_column]
    for column in numeric_columns:
        df = remove_outliers(df, column)
    print('6) count of valid rows:', df.count())
    process_times['Remove outliers'] = time.time() - step_start_time

    # Step 7: Remove duplicates from primary key column
    progress(0.8, desc="Removing duplicates from primary key")
    step_start_time = time.time()
    df = remove_duplicates_from_primary_key(df, primary_key_column)
    print('7) count of valid rows:', df.count())

    print("Cleaning process completed.")
    print_dataframe_info(df, "Final - ")

    return df, nonconforming_cells_before, process_times
