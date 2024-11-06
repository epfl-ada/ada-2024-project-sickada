import gzip
import json
import pandas as pd
import random

def random_sample_jsonl_rows(file_path, sample_size, random_seed=42):
    """
    Sample rows from a gzip-compressed JSON Lines file.

    This function reads a JSON Lines file and samples a specified number of rows randomly.
    It utilizes reservoir sampling to ensure that each row has an equal probability of being selected.

    Parameters:
    ----------
    file_path : str
        The path to the gzip-compressed JSON Lines file to sample from.
    sample_size : int
        The number of rows to sample from the file.
    random_seed : int, optional
        The seed for the random number generator to ensure reproducibility. Defaults to 42.

    Returns:
    -------
    pd.DataFrame
        A Pandas DataFrame containing the sampled rows from the JSON Lines file.
    """

    sampled_rows = []
    
    random.seed(random_seed)
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        total_lines = 0
        line_indices = []
        
        for line in f:
            total_lines += 1

            if total_lines <= sample_size:
                line_indices.append(line)
            else:
                index_to_replace = random.randint(0, total_lines - 1)
                if index_to_replace < sample_size:
                    line_indices[index_to_replace] = line

    sampled_rows = [json.loads(line) for line in line_indices[:sample_size]]     
    
    return pd.DataFrame(sampled_rows)
