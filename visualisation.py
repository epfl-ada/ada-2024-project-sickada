import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import ccf

def plot_time_series(df, topic_name, sample_size, save_dir='time_series_plots'):
    """
    Plots new content creation over time based on the provided DataFrame and topic name.

    Parameters:
    - df (DataFrame): 
        The DataFrame containing the content creation data.
    - topic_name (str): 
        The topic name to be included in the plot title and filename.
    - sample_size (int): 
        The sample size to be included in the plot title and filename.
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

    plt.figure(figsize=(12, 6))
    categories = pivot_data.columns
    x = np.arange(len(pivot_data))
    width = 0.2

    for i, category in enumerate(categories):
        plt.bar(x + i * width, pivot_data[category], width, label=category)

    plt.xlabel('Time Period')
    plt.ylabel('New Content Creation (s)')
    plt.title(f'{topic_name} Content Creation Over Time with {sample_size} samples from YouNiverse Dataset')
    plt.legend(title='Category')

    tick_spacing = 3
    plt.xticks(x[::tick_spacing], pivot_data.index.strftime('%m-%Y')[::tick_spacing], rotation=45)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f"{topic_name.replace(' ', '_')}_time_series_{sample_size}_samples.png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_cross_correlation(df, topic_name, sample_size, save_directory='cross_correlation_plots', max_lag=20):
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
    print(f"Cross-correlation plot saved as {file_path}")