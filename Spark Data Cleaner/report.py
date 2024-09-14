import os
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, count, when, lit, isnan
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType, StringType, DateType, TimestampType

REPORT_DIR = f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(REPORT_DIR, exist_ok=True)


def save_plot(fig, filename):
    fig.savefig(os.path.join(REPORT_DIR, filename), dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_heatmap(df, title):
    # Calculate the percentage of null values for each column
    null_percentages = df.select([
        (100 * count(when(col(c).isNull() | isnan(c), c)) / count('*')).alias(c)
        for c in df.columns
    ]).toPandas()

    plt.figure(figsize=(12, 8))
    sns.heatmap(null_percentages, cbar=True, cmap='Reds', annot=True, fmt='.1f')
    plt.title(title)
    plt.ylabel('Percentage of Missing Values')
    plt.tight_layout()
    save_plot(plt.gcf(), f'{title.lower().replace(" ", "_")}.png')


def plot_column_schemas(df):
    # Get the data types of all columns
    schema = df.schema
    data_types = []
    for field in schema.fields:
        dtype_name = field.dataType.typeName()
        print(f"Column '{field.name}' has data type '{dtype_name}'")
        data_types.append(dtype_name.capitalize())

    # Count the occurrences of each data type
    type_counts = Counter(data_types)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate a color palette with as many colors as there are bars
    colors = plt.cm.tab20(np.linspace(0, 1, len(type_counts)))

    # Plot the bars
    bars = ax.bar(type_counts.keys(), type_counts.values(), color=colors)

    ax.set_title('Column Data Types')
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Count')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, 'column_schemas.png')


def plot_nonconforming_cells(nonconforming_cells):
    # Ensure that nonconforming_cells is a dictionary
    if isinstance(nonconforming_cells, dict):
        # Proceed with plotting if it's a dictionary
        fig, ax = plt.subplots(figsize=(12, 6))

        # Generate a color palette with as many colors as there are bars
        colors = plt.cm.rainbow(np.linspace(0, 1, len(nonconforming_cells)))

        # Plot the bars
        bars = ax.bar(list(nonconforming_cells.keys()), list(nonconforming_cells.values()), color=colors)

        ax.set_title('Nonconforming Cells by Column')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Number of Nonconforming Cells')
        plt.xticks(rotation=90)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:,}',
                    ha='center', va='bottom')

        save_plot(fig, 'nonconforming_cells.png')
    else:
        print(f"Expected nonconforming_cells to be a dictionary, but got {type(nonconforming_cells)}.")


def plot_column_distributions(cleaned_df, primary_key_column):
    print("Plotting distribution charts for numeric columns in the cleaned DataFrame...")

    def get_numeric_columns(df):
        return [field.name for field in df.schema.fields
                if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType))
                and field.name != primary_key_column]

    numeric_columns = get_numeric_columns(cleaned_df)
    num_columns = len(numeric_columns)

    if num_columns == 0:
        print("No numeric columns found in the cleaned DataFrame for distribution plots.")
        return

    # Create subplots for distributions
    ncols = 3
    nrows = (num_columns + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten() if num_columns > 1 else [axes]

    for i, column in enumerate(numeric_columns):
        # Convert to pandas for plotting
        cleaned_data = cleaned_df.select(column).toPandas()[column].dropna()

        sns.histplot(cleaned_data, ax=axes[i], kde=True, color='orange', label='After Cleaning', alpha=0.7)
        axes[i].set_title(f'{column} - Distribution After Cleaning')
        axes[i].legend()

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_plot(fig, 'distributions_after_cleaning.png')


def plot_boxplot_with_outliers(original_df, primary_key_column):
    print("Plotting boxplots for numeric columns in the original DataFrame...")

    def get_numeric_columns(df):
        return [field.name for field in df.schema.fields
                if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType))
                and field.name != primary_key_column]

    numeric_columns = get_numeric_columns(original_df)
    num_columns = len(numeric_columns)

    if num_columns == 0:
        print("No numeric columns found in the original DataFrame for boxplots.")
        return

    # Create subplots based on the number of numeric columns
    ncols = 3
    nrows = (num_columns + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten() if num_columns > 1 else [axes]

    for i, column in enumerate(numeric_columns):
        # Convert data to pandas for plotting
        data = original_df.select(column).toPandas()[column].dropna()

        sns.boxplot(x=data, ax=axes[i], color='blue', orient='h')
        axes[i].set_title(f'Boxplot of {column} (Before Cleaning)')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_plot(fig, 'boxplots_before_cleaning.png')


def plot_correlation_heatmap(df, primary_key_column):
    # Select only numeric columns
    numeric_columns = [field.name for field in df.schema.fields
                       if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType))
                       and field.name != primary_key_column]

    if not numeric_columns:
        print("No numeric columns found for correlation heatmap.")
        return

    # Create a vector column of numeric columns
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
    df_vector = assembler.transform(df).select("features")

    # Compute correlation matrix
    matrix = Correlation.corr(df_vector, "features").collect()[0][0]
    corr_matrix = matrix.toArray().tolist()

    # Convert to pandas DataFrame for plotting
    corr_df = pd.DataFrame(corr_matrix, columns=numeric_columns, index=numeric_columns)

    # Plot the heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    save_plot(plt.gcf(), 'correlation_heatmap.png')


def plot_process_times(process_times):
    # Convert seconds to minutes
    process_times_minutes = {k: v / 60 for k, v in process_times.items()}

    # Separate main processes and column cleaning processes
    main_processes = {k: v for k, v in process_times_minutes.items() if not k.startswith("Clean column:")}
    column_processes = {k: v for k, v in process_times_minutes.items() if k.startswith("Clean column:")}

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot main processes
    bars1 = ax1.bar(main_processes.keys(), main_processes.values())
    ax1.set_title('Main Process Times')
    ax1.set_ylabel('Time (minutes)')
    ax1.tick_params(axis='x', rotation=45)

    # Plot column cleaning processes
    bars2 = ax2.bar(column_processes.keys(), column_processes.values())
    ax2.set_title('Column Cleaning Times')
    ax2.set_ylabel('Time (minutes)')
    ax2.tick_params(axis='x', rotation=90)

    # Add value labels on top of each bar
    for ax, bars in zip([ax1, ax2], [bars1, bars2]):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.8f}', ha='center', va='bottom')

    # Add total time to the plot
    total_time = sum(process_times_minutes.values())
    fig.suptitle(f'Process Times (Total: {total_time:.6f} minutes)', fontsize=16)

    plt.tight_layout()
    save_plot(fig, 'process_times.png')


def create_full_report(original_df, cleaned_df, nonconforming_cells_before, process_times, removed_columns,
                       removed_rows, primary_key_column):
    os.makedirs(REPORT_DIR, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 400

    print("Plotting nonconforming cells before cleaning...")
    plot_nonconforming_cells(nonconforming_cells_before)

    print("Plotting column distributions...")
    plot_column_distributions(cleaned_df, primary_key_column)

    print("Plotting boxplots for original data...")
    plot_boxplot_with_outliers(original_df, primary_key_column)

    print("Plotting process times...")
    plot_process_times(process_times)

    print("Plotting heatmaps...")
    plot_heatmap(original_df, "Missing Values Before Cleaning")

    print("Plotting correlation heatmap...")
    plot_correlation_heatmap(cleaned_df, primary_key_column)

    print("Plotting column schemas...")
    plot_column_schemas(cleaned_df)

    print(f"All visualization reports saved in directory: {REPORT_DIR}")