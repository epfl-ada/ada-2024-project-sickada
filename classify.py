import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from tqdm import tqdm
from datasets import Dataset
import ast


def classify_batch(batch, candidate_labels, on, classifier, multi_label=False):
    """Perform classification on a batch

    Args:
        batch (datasets.Dataset): Batch of data to be classified
        candidate_labels (list(str)): Labels used for classification
        on (str): Column to be classified
        classifier (transformers.pipeline): Classification pipeline
        multi_label (bool, optional): Decide whether the text is classified for each label independently or not. Defaults to False.

    Returns:
        datasets.Dataset: The original batch with a new column containing the labels.
    """

    texts = batch[on]
    results = classifier(texts, candidate_labels, multi_label=multi_label)

    labels_per_text = []
    for result in results:
        scores, labels = result["scores"], result["labels"]
        sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        scores, labels = zip(*sorted_pairs)

        max_score = scores[0]
        threshold = max_score * 0.9
        top_count = len([score for score in scores if score >= threshold])

        # Apply classification logic based on thresholds
        if max_score < 0.3:
            labels_per_text.append(["misc"])
        elif top_count == 1:
            labels_per_text.append([labels[0]])
        elif top_count == 2 and multi_label:
            labels_per_text.append([labels[0], labels[1]])
        elif top_count == 3 and multi_label:
            labels_per_text.append([labels[0], labels[1], labels[2]])
        else:
            labels_per_text.append(["uncertain"])

    # Add the classification labels as a new column in the batch
    batch["classified_labels"] = labels_per_text
    return batch


def classify(
    data, candidate_labels, classifier, on="text", multi_label=True, batch_size=16
):
    """Labels text from a pandas DataFrame

    Args:
        data (pd.Dataframe): DataFrame containing the text to be classified
        candidate_labels (list(str)): Labels used for classification
        classifier (transformers.pipeline): Classification pipeline
        on (str, optional): Column used for classification. Defaults to "text".
        multi_label (bool, optional): Decide whether the text is classified for each label independently or not. Defaults to True.
        batch_size (int, optional): Batch size. Defaults to 16.

    Returns:
        pd.DataFrame: The original DataFrame with a new column containing the labels
    """

    print("Converting to dataset...")
    dataset = Dataset.from_pandas(data)
    print("Processing...")
    dataset = dataset.map(
        classify_batch,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={
            "classifier": classifier,
            "candidate_labels": candidate_labels,
            "on": on,
            "multi_label": multi_label,
        },
    )
    print("Converting back to DataFrame...")
    final_data = dataset.to_pandas()

    # Convert `classified_labels` column from string to list if needed
    if multi_label:
        final_data["classified_labels"] = final_data["classified_labels"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    return final_data
