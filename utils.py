import os 
import os.path as op
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

from googleapiclient.discovery import build

API_KEY = pd.read_json(op.join('.','config.json'))['api_key'][0] # local file w personal API key



# ______________________________________________________________________________________________________________________
# Data extraction 
# ______________________________________________________________________________________________________________________

def filter_jsonl(input_path, category, batch_size, save_path, verbose = False):
    """Unzips input jsonl data then extracts rows with given category and saves them in batches

    Args:
        input (str): path to yt_metadata_en.jsonl.gz (incl)
        category (str): from the options in channel metadata
        batch_size (int): number of videos per batch
        save_path (str): path to folder where you want the batch dataframes to be saved
        verbose (bool, optional): print info. Defaults to False.
    """
    
    batch_index = -1 # so we can start with index 0
    line_counter = 0
    category_counter = 0
    renew_list = True # bc issue: 0 % anythig = 0
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            line_counter +=1
            
            # create new batch list
            if category_counter % batch_size == 0 and renew_list:
                renew_list = False
                filtered_data = []
                batch_index +=1
                if verbose:
                    print(f'======== Batch {batch_index} - started at {line_counter} ========')
            
            if entry.get('categories') == category:
                category_counter +=1
                filtered_data.append(entry)
                
                if verbose:
                    if category_counter != 0 and category_counter % 100000 == 0:
                        print(f'Filtered {category_counter} {category} videos out of {line_counter} so far') 
                
                if len(filtered_data) == batch_size: #save
                    df_filtered = pd.DataFrame(filtered_data)
                    df_filtered.to_csv(os.path.join(save_path, f'{category}_videos_{batch_index}.csv'))
                    renew_list = True
                    if verbose: 
                        print(f"We filtered a total of {category_counter} videos in the {category} category!")
        
        df_filtered = pd.DataFrame(filtered_data)
        df_filtered.to_csv(os.path.join(save_path, f'{category}_videos_{batch_index}.csv'))
        
        print(f"We filtered a total of {category_counter} videos in the {category} category!")


# ______________________________________________________________________________________________________________________
# BART classification functions
# ______________________________________________________________________________________________________________________


def load_metadata_videos(file_path):
    """
    Load the metadata of the videos from the file_path
    """
    return pd.read_csv(file_path).drop(columns='Unnamed: 0').dropna()

def bart_classification(text, candidate_labels, multi_label = True, plot=False, title=''):
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
    scores, labels = result['scores'], result['labels']
    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    scores, labels = zip(*sorted_pairs)

    max_score = scores[0]
    threshold = max_score * 0.9
    top_count = len([i for i, score in enumerate(scores) if score >= threshold])

    if plot: plot_scores_BART(scores, labels, top_count, title)

    if max_score < 0.3: return ["misc"]
    elif top_count == 1: return [labels[0]]
    elif top_count == 2 and multi_label: return [labels[0], labels[1]]
    elif top_count == 3 and multi_label: return [labels[0], labels[1], labels[2]]
    else: return ["uncertain"]


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
    colors = ['green' if i < top_count else 'grey' for i in range(len(labels))]

    # Map x-axis labels to integers from 1 to len(labels)
    x_positions = range(1, len(labels) + 1)

    # Create a figure with two subplots
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(12, 4))

    # Plot the main bar chart on the first subplot
    bars = ax_main.bar(x_positions, scores, color=colors)

    # Add score labels above each bar
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax_main.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,  # Slightly above the bar
            f'{score:.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Customize the main subplot
    ax_main.set_title(f'Probability of Each Label for the Video:\n{title}', fontsize=14)
    ax_main.set_xlabel('Label Numbers', fontsize=12)
    ax_main.set_ylabel('Probability', fontsize=12)
    ax_main.set_ylim(0, max(scores) + 0.1)  # Add some space on top for labels
    ax_main.set_xticks(x_positions)  # Use integers on x-axis
    ax_main.grid(axis='y', linestyle='--', alpha=0.7)

    # Set up the legend subplot with small points at (0,0) for each label
    for i, label in enumerate(labels):
        ax_legend.plot(0, 0, 'o', color='white', label=f"{i + 1}: {label}")  # White dot as a placeholder

    # Hide the legend subplot axes and only show the legend
    ax_legend.legend(loc='center', fontsize=9)
    ax_legend.axis('off')

    # Display the plot with tight layout
    plt.tight_layout()
    plt.show()


# ______________________________________________________________________________________________________________________
# Functions to extract countries from the education channels - YouTube API
# ______________________________________________________________________________________________________________________

def extract_channels_edu(path_edu, N_BATCHES, verbose = False):
    channels = []
    for i in range(N_BATCHES):
            if verbose :
                print(f'Processing file : path_edu_{i}', end = '')
            edu = pd.read_csv(path_edu.format(i), index_col=0)
            ch = list(pd.unique(edu['channel_id']))
            if verbose : 
                print(f"  --> Found {len(ch)} channels")
            channels.extend(ch)
    channels = list(set(channels)) # take unique of the junction
    if verbose:
         print('Total number of unique channels :' , len(channels))
    return channels

def agglomerate_countries(x, val_counts, filter = 10):
    if type(x) == str and val_counts[x] < filter:
        return 'Other'
    elif type(x) == str and x == 'deleted': # assign deleted to 'unknown'
        return '?'
    elif type(x) == float : # assign NaN to 'unknown'
        return '?'
    else:
        return x


def youtube_country_scraper(channel_ids, verbose = False):
    # Disable OAuthlib's HTTPS verification when running locally. *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    youtube = build('youtube', 'v3', developerKey = API_KEY)
    ids_string = ",".join(channel_ids)
    
    request = youtube.channels().list(
        part = 'snippet',
        id= ids_string
    )
    items = request.execute()
    countries = {ch: 'Redo' for ch in channel_ids}
    if ('items' in items): # for when you redo with single channels
        for item in items.get('items', []):
            if 'snippet' in item:
                id = item.get('id')
                country  = item.get('snippet').get('country')
                if (id in channel_ids): # else the channel now has a different id and need to be redone
                    countries[id] = country
            else:
                countries[id] = None
    else:
        countries[list(countries)[0]] = 'deleted' # channel info is not available anymore
    if verbose :
        print(items)
        print(countries)
    return countries



# ______________________________________________________________________________________________________________________
# Functions FRED
# ______________________________________________________________________________________________________________________

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

