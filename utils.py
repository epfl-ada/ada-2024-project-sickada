import pandas as pd
import glob
import random


def random_sample_from_csv_files(directory_path, total_sample_size):
    # Find all CSV files that start with "Education_videos_"
    file_paths = glob.glob(f"{directory_path}/Education_videos_*.csv")

    # Initialize an empty list for the reservoir
    reservoir = []
    row_index = 0  # Track the index of the current row across all files

    # Process each file one by one
    for file_path in file_paths:
        # Read the file in chunks to manage memory usage
        df = pd.read_csv(file_path)
        # Iterate over each row in the current file
        for _, row in df.iterrows():
            if len(reservoir) < total_sample_size:
                # If reservoir is not full, add the row directly
                reservoir.append(row)
            else:
                # If reservoir is full, replace an element with decreasing probability
                replace_index = random.randint(0, row_index)
                if replace_index < total_sample_size:
                    reservoir[replace_index] = row

            # Increment the global row index
            row_index += 1

    # Convert the reservoir (list of rows) back to a DataFrame
    sampled_df = pd.DataFrame(reservoir)

    return sampled_df


def create_channels_cat(df_cat: pd.DataFrame):
    # Count the number of categories for each channel ids
    channels_cat = (
        df_cat.groupby("channel_id")["broad_category"]
        .value_counts()
        .reset_index(name="count")
    )
    # Assign a weight to each category for each channel ids (count/total count)
    channels_cat["weights"] = channels_cat.groupby("channel_id")["count"].transform(
        lambda x: x / x.sum()
    )
    # Aggregate the categories and weights to lists
    result = (
        channels_cat.groupby("channel_id")
        .agg(categories=("broad_category", list), weights=("weights", list))
        .reset_index()
    )
    return result
