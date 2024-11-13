import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from tqdm import tqdm
from datasets import Dataset
import ast


def classify_dataset(batch, candidate_labels, on, classifier, multi_label=False):
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
    print("Converting to dataset...")
    dataset = Dataset.from_pandas(data)
    print("Processing...")
    dataset = dataset.map(
        classify_dataset,
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

    # Drop unnecessary columns
    final_data = final_data.drop(columns=["Unnamed: 0", "__index_level_0__"])

    # Convert `classified_labels` column from string to list if needed
    if multi_label:
        final_data["classified_labels"] = final_data["classified_labels"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    return final_data
