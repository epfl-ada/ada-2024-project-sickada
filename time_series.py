import pandas as pd

def create_time_slots(df, frequency='ME', max_non_zero_count=10):
    """
    Discretize video upload dates into time slots with formatted names, capturing video duration in corresponding slots,
    and filtering out categories with more than a specified count of non-zero entries.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing video data with 'upload_date', 'display_id', 'categories', and 'duration' columns.
    frequency : str, optional
        Frequency for time slot creation: 'W' for weekly or 'ME' for monthly end. Default is 'ME' (monthly end).
    max_non_zero_count : int, optional
        Maximum number of non-zero entries allowed in the time series for a category to be retained in the output.

    Returns:
    -------
    pd.DataFrame
        DataFrame with aggregated duration values for each category across defined time slots, 
        with columns named by date and filtered according to `max_non_zero_count`.
    """
    # Convert 'upload_date' to datetime and handle potential NaT values
    df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
    
    # Drop rows where 'upload_date' is NaT and create a copy to avoid SettingWithCopyWarning
    df = df.dropna(subset=['upload_date']).copy()

    #df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
    #df = df.dropna(subset=['upload_date'])

    if df['upload_date'].empty:
        raise ValueError("The DataFrame does not contain valid 'upload_date' values.")

    start_date = df['upload_date'].min()
    end_date = df['upload_date'].max()

    if frequency == 'W':
        adjusted_start_date = start_date - pd.DateOffset(weeks=1)
        adjusted_end_date = end_date + pd.DateOffset(weeks=1)
        freq_format = "%U-%Y"  # Week-YYYY format
    elif frequency == 'ME':
        adjusted_start_date = start_date - pd.DateOffset(months=1)
        adjusted_end_date = end_date + pd.DateOffset(months=1)
        freq_format = "%m-%Y"  # MM-YYYY format
    else:
        adjusted_start_date = start_date - pd.Timedelta(days=1)
        adjusted_end_date = end_date + pd.Timedelta(days=1)
        freq_format = "%m-%Y"

    time_slots = pd.date_range(start=adjusted_start_date, end=adjusted_end_date, freq=frequency)
    time_slot_labels = [date.strftime(freq_format) for date in time_slots]

    result_df = pd.DataFrame(columns=['video_id', 'category'] + time_slot_labels)

    for index, row in df.iterrows():
        video_id = row['display_id']
        category = row['categories']
        upload_date = row['upload_date']
        
        upload_time_slot_index = (upload_date - time_slots[0]).days // (7 if frequency == 'W' else 30)

        if 0 <= upload_time_slot_index < len(time_slots):
            new_row = [video_id, category] + [0] * len(time_slots)
            new_row[2 + upload_time_slot_index] = row['duration']
            result_df.loc[len(result_df)] = new_row

    aggregated_df = result_df.groupby('category').sum().reset_index()

    filtered_df = aggregated_df[aggregated_df.apply(lambda row: (row != 0).sum() > max_non_zero_count, axis=1)]
    
    return filtered_df