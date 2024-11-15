import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op
from statsmodels.tsa.stattools import ccf
import matplotlib.lines as mlines

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