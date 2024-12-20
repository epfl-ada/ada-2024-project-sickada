import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
from cycler import cycler

def rolling_dtw_analysis(time_series_df, topic, target_series_name='Education', window_size=3, save_path=None):
    """
    Perform rolling DTW analysis between a target series and all other series in a DataFrame.

    Parameters:
    - time_series_df (pd.DataFrame): DataFrame with time series data, rows as series and columns as time points.
    - target_series_name (str): The name of the target series to compare against.
    - window_size (int): Window size for rolling DTW computation.
    - save_path (str, optional): Path to save the plot. If None, the plot is displayed but not saved.

    Returns:
    - rolling_dtw_results (dict): Dictionary with categories as keys and their DTW distances as values.
    - plt.Figure: A plot showing DTW distances over time.
    """
    def rolling_dtw(series, target_series, window_size):
        dtw_distances = []
        for i in range(len(series) - window_size + 1):
            segment = series[i:i + window_size]
            target_segment = target_series[i:i + window_size]
            distance = euclidean(segment, target_segment)
            dtw_distances.append(distance)
        return dtw_distances

    Normaize = True
    if Normaize:
        scaler = StandardScaler()
        normalized_df = pd.DataFrame(
            
            scaler.fit_transform(time_series_df.values.T),
            index=time_series_df.columns,
            columns=time_series_df.index
        )
        normalized_df = normalized_df.T
    else: 
        normalized_df = time_series_df
        
    target_series = normalized_df.loc[target_series_name].values

    rolling_dtw_results = {}

    for category in time_series_df.index:
        if category != target_series_name:
            series = normalized_df.loc[category].values
            rolling_dtw_result = rolling_dtw(series, target_series, window_size=window_size)
            #time_index = [pd.Timestamp(col).strftime('%Y-%m') for col in normalized_df.columns[:len(series) - window_size + 1]]
            time_index = ['06-2015', '08-2015', '10-2015', '12-2015', 
                '02-2016', '04-2016', '06-2016', '08-2016',
                '10-2016', '12-2016', '02-2017', '04-2017',
                '06-2017', '08-2017', '10-2017', '12-2017',
                '02-2018', '04-2018', '06-2018', '08-2018']
            rolling_dtw_results[category] = pd.Series(rolling_dtw_result, index=time_index)

    plt.figure(figsize=(14, 7))
    
    colors = plt.cm.tab10.colors[1:]
    plt.gca().set_prop_cycle(cycler(color=colors))

    for category, dtw_series in rolling_dtw_results.items():
        plt.plot(dtw_series, label=category)

    plt.xlabel('Time (looking at {} period time)'.format(window_size))
    plt.ylabel('Normalized DTW Distance')
    plt.title(f'Dynamic Time Warping of {target_series_name} with other categories for topic: {topic}')
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.tight_layout()

    if save_path:
        if not os.path.exists('data/figures/causality'):
            os.makedirs('data/figures/causality')
        plt.savefig(os.path.join(save_path))

    return rolling_dtw_results



def granger_causality_analysis(time_series_df, target_series_name='Education', max_lag=12, save_path=None):
    def granger_causality_test(series, target_series, max_lag):
        series_df = pd.DataFrame({
            'target_series': target_series.values,
            'cause_series': series.values
        })
        test_result = grangercausalitytests(series_df[['target_series', 'cause_series']], maxlag=max_lag, verbose=False)
        p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
        return p_values

    target_series = time_series_df.loc[target_series_name]

    granger_results = {}

    for category in time_series_df.index:
        if category != target_series_name:
            category_series = time_series_df.loc[category]
            p_values = granger_causality_test(category_series, target_series, max_lag=max_lag)
            granger_results[category] = p_values

    results_df = pd.DataFrame.from_dict(granger_results, orient='index', columns=[f'Lag {i+1}' for i in range(max_lag)])

    plt.figure(figsize=(8, 6))
    sns.heatmap(results_df, annot=True, fmt='.4f', cmap='coolwarm', cbar_kws={'label': 'p-value'})
    plt.title(f'Granger Causality Results for {target_series_name}')
    plt.xlabel('Lags')
    plt.ylabel('Categories')

    if save_path:
        if not os.path.exists('data/figures/causality'):
            os.makedirs('data/figures/causality')
        plt.savefig(os.path.join(save_path))

    #plt.show()

    return granger_results