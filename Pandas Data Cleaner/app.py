import gradio as gr
import pandas as pd
import os
from datetime import datetime
from clean import clean_data, get_numeric_columns
from report import create_full_report, REPORT_DIR

def clean_and_visualize(file, primary_key_column, progress=gr.Progress()):
    # Read the CSV file
    progress(0.05, desc="Reading CSV file")
    df = pd.read_csv(file.name)

    # Clean the data
    progress(0.1, desc="Starting data cleaning")
    cleaned_df, nonconforming_cells_before, process_times = clean_data(df, primary_key_column, progress)
    progress(0.8, desc="Data cleaning completed")

    # Calculate removed columns and rows
    removed_columns = len(df.columns) - len(cleaned_df.columns)
    removed_rows = len(df) - len(cleaned_df)

    # Generate full visualization report
    progress(0.9, desc="Generating report")
    create_full_report(
        df,
        cleaned_df,
        nonconforming_cells_before,
        process_times,
        removed_columns,
        removed_rows,
        primary_key_column
    )

    # Save cleaned data as CSV
    progress(0.95, desc="Saving cleaned data")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cleaned_csv_path = os.path.join(f"cleaned_data_{current_time}.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False)

    # Collect all generated images
    image_files = [os.path.join(REPORT_DIR, f) for f in os.listdir(REPORT_DIR) if f.endswith('.png')]

    progress(1.0, desc="Process completed")
    return cleaned_csv_path, image_files

def launch_app():
    with gr.Blocks() as app:
        gr.Markdown("# Data Cleaner")

        with gr.Row():
            file_input = gr.File(label="Upload CSV File", file_count="single", file_types=[".csv"])

        with gr.Row():
            primary_key_dropdown = gr.Dropdown(label="Select Primary Key Column", choices=[], interactive=True)

        with gr.Row():
            clean_button = gr.Button("Start Cleaning")

        with gr.Row():
            progress_bar = gr.Progress()

        with gr.Row():
            cleaned_file_output = gr.File(label="Cleaned CSV", visible=True)

        with gr.Row():
            output_gallery = gr.Gallery(
                label="Visualization Results",
                show_label=True,
                elem_id="gallery",
                columns=[3],
                rows=[3],
                object_fit="contain",
                height="auto",
                visible=False
            )

        def update_primary_key_options(file):
            if file is None:
                return gr.Dropdown(choices=[])

            df = pd.read_csv(file.name)
            numeric_columns = get_numeric_columns(df)

            return gr.Dropdown(choices=numeric_columns)

        def process_and_show_results(file, primary_key_column):
            cleaned_csv_path, image_files = clean_and_visualize(file, primary_key_column, progress=progress_bar)
            return (
                cleaned_csv_path,
                gr.Gallery(visible=True, value=image_files)
            )

        file_input.change(
            fn=update_primary_key_options,
            inputs=file_input,
            outputs=primary_key_dropdown
        )

        clean_button.click(
            fn=process_and_show_results,
            inputs=[file_input, primary_key_dropdown],
            outputs=[cleaned_file_output, output_gallery]
        )

    app.launch()

if __name__ == "__main__":
    launch_app()