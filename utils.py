import pandas as pd
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_weighted_timeseries(
    df_ts: pd.DataFrame, channels_cat: pd.DataFrame, category: str, year: int
):
    # Filter the channels and extract category weight
    filtered_df = category_filter(channels_cat, category)
    # Keep channels that are in common with the classified videos channels
    df_ts_filtered = df_ts[df_ts["channel"].isin(filtered_df["channel_id"])]

    # Merge to bring in category weights
    df_ts_weighted = df_ts_filtered.merge(
        filtered_df[["channel_id", "category_weight"]],
        left_on="channel",
        right_on="channel_id",
        how="left",
    )

    # Apply weights to metrics
    df_ts_weighted["weighted_delta_views"] = (
        df_ts_weighted["delta_views"] * df_ts_weighted["category_weight"]
    )
    df_ts_weighted["weighted_delta_videos"] = (
        df_ts_weighted["delta_videos"] * df_ts_weighted["category_weight"]
    )
    df_ts_weighted["weighted_delta_subs"] = (
        df_ts_weighted["delta_subs"] * df_ts_weighted["category_weight"]
    )

    # Plot delta views, delta videos, delta subs, weighted by channel category weights
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax = ax.flatten()
    peak_views = (
        df_ts_weighted[df_ts_weighted["year"] == year]
        .groupby("month")["weighted_delta_views"]
        .sum()
    )
    sns.barplot(peak_views, ax=ax[0])
    ax[0].set_ylabel("Weighted Delta Views")
    ax[0].set_title(f"Weighted delta views, Category {category}, Year {year}")

    peak_uploads = (
        df_ts_weighted[df_ts_weighted["year"] == year]
        .groupby("month")["weighted_delta_videos"]
        .sum()
    )
    sns.barplot(peak_uploads, ax=ax[1])
    ax[1].set_ylabel("Weighted Delta Videos")
    ax[1].set_title(f"Weighted delta videos, Category {category}, Year {year}")

    peaks_subs = (
        df_ts_weighted[df_ts_weighted["year"] == year]
        .groupby("month")["weighted_delta_subs"]
        .sum()
    )
    sns.barplot(peaks_subs, ax=ax[2])
    ax[2].set_ylabel("Weighted Delta Subscribers")
    ax[2].set_title(f"Weighted delta subscribers, Category {category}, Year {year}")

    plt.tight_layout()


def plot_unweighted_timeseries(
    df_ts: pd.DataFrame, channels_cat: pd.DataFrame, category: str, year: int
):
    # Filter the channels and extract category weight
    filtered_df = category_filter(channels_cat, category)
    # Keep channels that are in common with the classified videos channels
    df_ts_filtered = df_ts[df_ts["channel"].isin(filtered_df["channel_id"])]

    # Plot delta views, delta videos, delta subs, weighted by channel category weights
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax = ax.flatten()
    peak_views = (
        df_ts_filtered[df_ts_filtered["year"] == year]
        .groupby("month")["delta_views"]
        .sum()
    )
    sns.barplot(peak_views, ax=ax[0])
    ax[0].set_ylabel("Delta Views")
    ax[0].set_title(f"Unweighted delta views, Category {category}, Year {year}")

    peak_uploads = (
        df_ts_filtered[df_ts_filtered["year"] == year]
        .groupby("month")["delta_videos"]
        .sum()
    )
    sns.barplot(peak_uploads, ax=ax[1])
    ax[1].set_ylabel("Delta Videos")
    ax[1].set_title(f"Unweighted delta videos, Category {category}, Year {year}")

    peaks_subs = (
        df_ts_filtered[df_ts_filtered["year"] == year]
        .groupby("month")["delta_subs"]
        .sum()
    )
    sns.barplot(peaks_subs, ax=ax[2])
    ax[2].set_ylabel("Delta Subscribers")
    ax[2].set_title(f"Unweighted delta subscribers, Category {category}, Year {year}")

    plt.tight_layout()


def plot_timeseries_single_category(
    df_ts: pd.DataFrame, channels_cat: pd.DataFrame, category: str, year: int
):

    filtered_df = channels_cat[channels_cat["dominant_category"] == category]

    df_ts_cat = df_ts[df_ts["channel"].isin(filtered_df["channel_id"])]
    df_ts_cat = df_ts_cat[df_ts_cat["year"] == year]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax = ax.flatten()
    peak_views = df_ts_cat.groupby("month")["delta_views"].sum()
    sns.barplot(peak_views, ax=ax[0])
    ax[0].set_ylabel("Delta Views")
    ax[0].set_title(f"delta views, Category {category}, Year {year}")

    peak_videos = df_ts_cat.groupby("month")["delta_videos"].sum()
    sns.barplot(peak_videos, ax=ax[1])
    ax[1].set_ylabel("Delta Videos")
    ax[1].set_title(f"delta videos, Category {category}, Year {year}")

    peak_subs = df_ts_cat.groupby("month")["delta_subs"].sum()
    sns.barplot(peak_subs, ax=ax[2])
    ax[2].set_ylabel("Delta Subscribers")
    ax[2].set_title(f"delta subscribers, Category {category}, Year {year}")

    plt.tight_layout()
