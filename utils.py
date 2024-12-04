import os
import os.path as op

import pandas as pd
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import gzip

from transformers import pipeline


# ______________________________________________________________________________________________________________________
# Data extraction
# ______________________________________________________________________________________________________________________


def filter_jsonl(input_path, category, batch_size, save_path, verbose=False):
    """Unzips input jsonl data then extracts rows with given category and saves them in batches

    Args:
        input (str): path to yt_metadata_en.jsonl.gz (incl)
        category (str): from the options in channel metadata
        batch_size (int): number of videos per batch
        save_path (str): path to folder where you want the batch dataframes to be saved
        verbose (bool, optional): print info. Defaults to False.
    """

    batch_index = -1  # so we can start with index 0
    line_counter = 0
    category_counter = 0
    renew_list = True  # bc issue: 0 % anythig = 0
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            line_counter += 1

            # create new batch list
            if category_counter % batch_size == 0 and renew_list:
                renew_list = False
                filtered_data = []
                batch_index += 1
                if verbose:
                    print(
                        f"======== Batch {batch_index} - started at {line_counter} ========"
                    )

            if entry.get("categories") == category:
                category_counter += 1
                filtered_data.append(entry)

                if verbose:
                    if category_counter != 0 and category_counter % 100000 == 0:
                        print(
                            f"Filtered {category_counter} {category} videos out of {line_counter} so far"
                        )

                if len(filtered_data) == batch_size:  # save
                    df_filtered = pd.DataFrame(filtered_data)
                    df_filtered.to_csv(
                        os.path.join(save_path, f"{category}_videos_{batch_index}.csv")
                    )
                    renew_list = True
                    if verbose:
                        print(
                            f"We filtered a total of {category_counter} videos in the {category} category!"
                        )

        df_filtered = pd.DataFrame(filtered_data)
        df_filtered.to_csv(
            os.path.join(save_path, f"{category}_videos_{batch_index}.csv")
        )

        print(
            f"We filtered a total of {category_counter} videos in the {category} category!"
        )


# ______________________________________________________________________________________________________________________
# BART classification functions
# ______________________________________________________________________________________________________________________


def load_metadata_videos(file_path):
    """
    Load the metadata of the videos from the file_path
    """
    return pd.read_csv(file_path).drop(columns="Unnamed: 0").dropna()


def bart_classification(text, candidate_labels, multi_label=True, plot=False, title=""):
    """
    Perform zero-shot classification using BART model
    Parameters:
    - text: the text to classify
    - candidate_labels: the list of labels to classify the text into
    - multi_label: whether to allow multiple labels or not
    - plot: whether to plot the scores or not
    - title: the title of the plot
    Returns:
    - a list of labels
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels, multi_label=multi_label)
    scores, labels = result["scores"], result["labels"]
    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    scores, labels = zip(*sorted_pairs)

    max_score = scores[0]
    threshold = max_score * 0.9
    top_count = len([i for i, score in enumerate(scores) if score >= threshold])

    if plot:
        plot_scores_BART(scores, labels, top_count, title)

    if max_score < 0.3:
        return ["misc"]
    elif top_count == 1:
        return [labels[0]]
    elif top_count == 2 and multi_label:
        return [labels[0], labels[1]]
    elif top_count == 3 and multi_label:
        return [labels[0], labels[1], labels[2]]
    else:
        return ["uncertain"]


def plot_scores_BART(scores, labels, top_count, title):
    """
    Plot the scores of the labels of the BART classification
    Parameters:
    - scores: the scores of the labels
    - labels: the labels
    - top_count: the number of top labels
    - title: the title of the plot
    """
    # Define colors: green for the top scores, grey for others
    colors = ["green" if i < top_count else "grey" for i in range(len(labels))]

    # Map x-axis labels to integers from 1 to len(labels)
    x_positions = range(1, len(labels) + 1)

    # Create a figure with two subplots
    fig, (ax_main, ax_legend) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(12, 4)
    )

    # Plot the main bar chart on the first subplot
    bars = ax_main.bar(x_positions, scores, color=colors)

    # Add score labels above each bar
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax_main.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,  # Slightly above the bar
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Customize the main subplot
    ax_main.set_title(f"Probability of Each Label for the Video:\n{title}", fontsize=14)
    ax_main.set_xlabel("Label Numbers", fontsize=12)
    ax_main.set_ylabel("Probability", fontsize=12)
    ax_main.set_ylim(0, max(scores) + 0.1)  # Add some space on top for labels
    ax_main.set_xticks(x_positions)  # Use integers on x-axis
    ax_main.grid(axis="y", linestyle="--", alpha=0.7)

    # Set up the legend subplot with small points at (0,0) for each label
    for i, label in enumerate(labels):
        ax_legend.plot(
            0, 0, "o", color="white", label=f"{i + 1}: {label}"
        )  # White dot as a placeholder

    # Hide the legend subplot axes and only show the legend
    ax_legend.legend(loc="center", fontsize=9)
    ax_legend.axis("off")

    # Display the plot with tight layout
    plt.tight_layout()
    plt.show()



# ______________________________________________________________________________________________________________________
# Functions FRED
# ______________________________________________________________________________________________________________________


def remove_nan(df: pd.DataFrame):
    df = df.dropna()
    return df


def clean_non_ascii(text):
    return re.sub(r"[^\x00-\x7F]+", " ", text)


def replace_non_ascii_in_dataframe(df, columns=["title", "tags", "description"]):
    for column in columns:
        df.loc[:, column] = df[column].apply(
            lambda x: clean_non_ascii(x) if isinstance(x, str) else x
        )
    return df


def remove_rows_with_empty_strings(df, columns=["title", "tags", "description"]):
    # Filter out rows where any specified column has an empty string
    df_filtered = df[
        ~df[columns].apply(lambda row: any(cell == "" for cell in row), axis=1)
    ]
    return df_filtered


def clean_data(df: pd.DataFrame):
    out = remove_nan(df)
    out = replace_non_ascii_in_dataframe(out)
    out = remove_rows_with_empty_strings(out)
    return out


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


def create_channel_cat_single(df_cat: pd.DataFrame):
    # Assign channels to the category with the highest number of videos
    channel_cat = df_cat.groupby(["channel_id", "broad_category"]).size()
    channel_cat = (
        channel_cat.groupby("channel_id")
        .idxmax()
        .apply(lambda x: x[1])
        .reset_index(name="dominant_category")
    )
    return channel_cat


def category_filter(df: pd.DataFrame, category: str):
    # Return the dataframe filtered with the choosen category, and with the weight of the corresponding category
    filtered_df = df[df["categories"].apply(lambda x: category in x)]
    filtered_df["category_weight"] = filtered_df.apply(
        lambda row: row["weights"][row["categories"].index(category)], axis=1
    )
    return filtered_df


def add_datetime_info(df, column):
    df[column] = pd.to_datetime(df[column])
    df["month"] = df[column].dt.month
    df["year"] = df[column].dt.year
    df["day"] = df[column].dt.day
    return df
