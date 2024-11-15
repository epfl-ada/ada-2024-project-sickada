import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as op
from statsmodels.tsa.stattools import ccf
import matplotlib.lines as mlines

from utils import *

figures_path = op.join('data', 'figures')


def plot_time_series(df, topic_name, sample_size, frequency, event_dict=None, save_dir='time_series_plots'):
    """
    Plots new content creation over time based on the provided DataFrame and topic name,
    and marks major events with vertical lines and numbered labels only in the legend box.

    Parameters:
    - df (DataFrame): 
        The DataFrame containing the content creation data.
    - topic_name (str): 
        The topic name to be included in the plot title and filename.
    - sample_size (int): 
        The sample size to be included in the plot title and filename.
    - frequency (str): 
        Frequency of the time series ('W' for weekly, 'ME' for monthly).
    - event_dict (dict, optional): 
        A dictionary where keys are event dates (as strings or datetime objects) and values are event names.
    - save_dir (str, optional): 
        Directory where the plot will be saved. Default is 'time_series_plots'.

    Returns:
    - None: This function saves the plot to the specified directory.
    """

    time_slot_columns = []
    for col in df.columns:
        try:
            pd.to_datetime(col, format='%m-%Y')
            time_slot_columns.append(col)
        except ValueError:
            continue

    long_data = pd.melt(df, id_vars=['category'], value_vars=time_slot_columns, var_name='time_slot', value_name='content_creation')
    long_data['time_slot'] = pd.to_datetime(long_data['time_slot'], format='%m-%Y', errors='coerce')

    non_zero_data = long_data[long_data['content_creation'] > 0]
    if not non_zero_data.empty:
        min_time = non_zero_data['time_slot'].min()
        max_time = non_zero_data['time_slot'].max()
        long_data = long_data[(long_data['time_slot'] >= min_time) & (long_data['time_slot'] <= max_time)]

    pivot_data = long_data.pivot(index='time_slot', columns='category', values='content_creation').fillna(0)

    fig, ax1 = plt.subplots(figsize=(16, 6))

    categories = pivot_data.columns
    x = np.arange(len(pivot_data))
    width = 0.1

    # Plot categories with bars
    category_bars = []
    for i, category in enumerate(categories):
        bar = ax1.bar(x + i * width, pivot_data[category], width, label=category)
        category_bars.append(bar)

    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('New Content Creation (s)')
    ax1.set_title(f'{topic_name} Content Creation Over Time with {sample_size} samples from YouNiverse Dataset')

    tick_spacing = 3 if frequency == 'ME' else 4
    
    if frequency == 'ME':  # Monthly data
        date_format = '%m-%Y'  # Example: "01-2020"
    elif frequency == 'W':  # Weekly data
        date_format = '%U-%Y'  # Example: "01-2020" (week number and year)
    
    ax1.set_xticks(x[::tick_spacing])
    ax1.set_xticklabels(pivot_data.index.strftime(date_format)[::tick_spacing], rotation=45)

    event_labels = {}
    event_lines = []  # To store event line objects for legend
    if event_dict:
        event_counter = 1
        for event_date, event_name in event_dict.items():
            event_date = pd.to_datetime(event_date)

            if event_date in pivot_data.index:
                event_x_pos = np.where(pivot_data.index == event_date)[0][0]
                
                # Plot event line
                line = ax1.axvline(event_x_pos, color='g', linestyle='--', lw=2)  # Vertical line for event
                event_lines.append(line)
                
                event_labels[event_counter] = f"Event {event_counter}: {event_name}"
                event_counter += 1

    ax1.set_yscale('log')

    # Customize legend: First the categories, then the events
    category_legend_labels = [patch.get_label() for patch in category_bars]  # Bar labels for categories
    event_legend_labels = [label for label in event_labels.values()]  # Event labels

    # Create legend handles for the categories and events
    # Creating legend entries for events as lines
    event_handles = [mlines.Line2D([0], [0], color='g', linestyle='--', lw=2) for _ in event_lines]

    # Combine category and event handles/labels
    handles = category_bars + event_handles
    labels = category_legend_labels + event_legend_labels

    # Place the legend under the plot
    ax1.legend(handles, labels, title="Legend", loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    save_dir = os.path.join(figures_path, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frequency_label = 'monthly' if frequency == 'ME' else 'weekly'
    if event_dict is not None:
        file_suffix = "_with_events"
    else:
        file_suffix = "_without_events"

    file_path = os.path.join(save_dir, f"{topic_name.replace(' ', '_')}_time_series_{sample_size}_samples_{frequency_label}{file_suffix}.png")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_cross_correlation(df, topic_name, sample_size, save_directory='cross_correlation_plots', max_lag=50):
    """
    Plots the cross-correlation of different categories against the 'Education' category using bar plots.

    Parameters:
    - df (pd.DataFrame): 
        DataFrame containing the data with a 'category' column and time slots.
    - topic_name (str): 
        A name to include in the plot title and saved file name.
    - sample_size (int): 
        The sample size to be included in the plot title and filename.
    - save_directory (str): 
        Directory where the plot will be saved. It will be created if it doesn't exist.
    - max_lag (int): 
        The maximum number of lags for which to calculate cross-correlation.

    Returns:
    - None: This function saves the plot to the specified directory.
    """
    save_directory = os.path.join(figures_path, save_directory)
    os.makedirs(save_directory, exist_ok=True)

    time_slot_columns = [col for col in df.columns if col != 'category' and col != 'video_id']

    y_education = df.loc[df['category'] == 'Education', time_slot_columns].values.flatten()

    if y_education.size == 0:
        print("No data found for the 'Education' category. Skipping plot generation.")
        return

    cross_corr_results_list = []

    for index, row in df.iterrows():
        if row['category'] == 'Education':
            continue

        y_other = row[time_slot_columns].values.flatten()

        if y_other.size == 0:
            print(f"No data found for the category '{row['category']}'. Skipping this category.")
            continue

        cross_corr = ccf(y_education, y_other)[:max_lag]

        for lag in range(len(cross_corr)):
            cross_corr_results_list.append({
                'Category': row['category'], 
                'Lag': lag + 1,
                'Cross-Correlation': cross_corr[lag]
            })

    cross_corr_results = pd.DataFrame(cross_corr_results_list)

    if cross_corr_results.empty:
        print("No cross-correlation data to plot.")
        return

    plt.figure(figsize=(14, 8))
    
    bar_width = 0.15  
    lags = np.arange(1, max_lag + 1)
    
    for i, category in enumerate(cross_corr_results['Category'].unique()):
        category_data = cross_corr_results[cross_corr_results['Category'] == category]
        plt.bar(lags + i * bar_width, category_data['Cross-Correlation'], width=bar_width, label=category)

    plt.title(f'Cross-Correlation of Education with other categories - {topic_name}, {sample_size} samples from YouNiverse Dataset')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(ticks=lags + (len(cross_corr_results['Category'].unique()) - 1) * bar_width / 2, labels=lags)

    plt.tight_layout()

    file_path = os.path.join(save_directory, f'cross_correlation_with_{topic_name.replace(" ", "_")}_{sample_size}_samples.png')
    plt.savefig(file_path)
    plt.close()

# ______________________________________________________________________________________________________________________
# Functions Timeseries analysis
# ______________________________________________________________________________________________________________________
  
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

def plot_categories(drive_path, channels_df, categories):

    # load educational files
    educational_dfs_names = os.listdir(os.path.join(drive_path, 'extracted_Education'))
    edu_dfs = [pd.read_csv(os.path.join(drive_path, 'extracted_Education', name)) for name in educational_dfs_names]
    category_map = channels_df.set_index('channel')['category_cc'].to_dict()
    for edu_df in edu_dfs:
        edu_df['category_cc_mapped'] = edu_df['channel_id'].map(category_map)

    # hold category counts
    counts_df = pd.DataFrame(index=categories)

    # calculate counts for each dataframe
    for i, edu_df in enumerate(edu_dfs):
        counts = edu_df['category_cc_mapped'].value_counts(dropna=False).reindex(categories, fill_value=0)
        counts_df[f'df_{i}'] = counts

    # also count nans
    nan_counts = [edu_df['category_cc_mapped'].isna().sum() for edu_df in edu_dfs]
    counts_df.loc['NaN'] = nan_counts

    # sum to have the total of categories
    counts_df['total'] = counts_df.sum(axis=1)
    counts_df_sorted = counts_df.sort_values(by='total', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='total', y=counts_df_sorted.index, data=counts_df_sorted, palette='viridis')
    plt.title('Total Counts per Category (Ordered)')
    plt.xscale('log')
    plt.xlabel('Total Count')
    plt.ylabel('Category')
    plt.show()

def plot_educational_distribution(drive_path, categories):
    # load educational files
    educational_dfs_names = os.listdir(os.path.join(drive_path, 'extracted_Education'))
    edu_dfs = [pd.read_csv(os.path.join(drive_path, 'extracted_Education', name)) for name in educational_dfs_names]

    # count the number of educational videos per channel and group them by channel category
    total_channel_edu_df = pd.DataFrame()
    for edu_df in edu_dfs:  
        channel_grouped_edu_df = edu_df['channel_id'].value_counts().reset_index()
        unique_channels = edu_df[['channel_id', 'category_cc_mapped']].drop_duplicates(subset='channel_id')
        channel_grouped_edu_df = channel_grouped_edu_df.merge(unique_channels, on='channel_id', how='left')

        total_channel_edu_df = pd.concat([total_channel_edu_df, channel_grouped_edu_df])
    final_channel_df = total_channel_edu_df

    # subplot grid (adjust rows/cols based on the number of categories)
    categories.sort()
    num_categories = len(categories)
    cols = 4
    rows = (num_categories + cols - 1) // cols  # rows needed for given columns

    # Create a figure with subplots for each category
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharex=False, sharey=False)

    # Flatten the axes array to easily iterate and plot
    axes = axes.flatten()

    # Loop through each category to plot
    for i, category in enumerate(categories):
        if category != 'NaN':
            cat_data = final_channel_df[final_channel_df['category_cc_mapped']== category]
            cat_data = cat_data.rename(columns = {'count': 'video_count'})
            cat_data = cat_data.groupby(['channel_id', 'category_cc_mapped'], as_index = False)['video_count'].sum()
        else:
            cat_data = final_channel_df[final_channel_df['category_cc_mapped'].isna()]
            cat_data = cat_data.rename(columns = {'count': 'video_count'})
            cat_data = cat_data.groupby(['channel_id'], as_index = False)['video_count'].sum()


        # Plot the distribution
        sns.boxplot(data = cat_data, y='video_count', color='hotpink', ax = axes[i])
        axes[i].set_title(f'{category} channels - total = {len(cat_data)}')
        axes[i].set_ylabel('Number of videos [log]')
        axes[i].set_yscale('log')
            
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and display plot
    plt.suptitle('Educational Video Count Distribution in YouTube Channels', fontsize = 30)
    plt.tight_layout()
    plt.show()