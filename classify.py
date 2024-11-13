import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from tqdm import tqdm
from datasets import Dataset


# Function to classify a batch of texts
def classify_batch(texts, candidate_labels, classifier):
    return classifier(texts, candidate_labels=candidate_labels)


def classify_fine(data, categories, on, classifier, batch_size=16):
    # Step 2: Classify subcategories in batches based on assigned broad category
    subcategory_results = []
    # Classify subcategories in batches for this broad category
    for broad_category in categories:
        subcategories = categories[broad_category]
        subset_data = data[data["broad_category"] == broad_category]
        if subset_data.empty:
            continue  # Skip if there are no rows for this broad category

        subcategory_results = []
        for i in tqdm(
            range(0, len(subset_data), batch_size), desc=f"Processing {broad_category}"
        ):
            batch_texts = subset_data[on].iloc[i : i + batch_size].tolist()
            batch_results = classify_batch(batch_texts, subcategories, classifier)
            subcategory_results.extend(
                [(res["labels"][0], res["scores"][0]) for res in batch_results]
            )

        (
            data.loc[subset_data.index, "subcategory"],
            data.loc[subset_data.index, "subcategory_confidence"],
        ) = zip(*subcategory_results)
    return data


def classify_broad_dataset(batch, categories, on, classifier):
    texts = batch[on]
    broad_categories_labels = list(categories.keys())
    results = classifier(texts, broad_categories_labels)
    batch["broad_category"] = [res["labels"][0] for res in results]
    batch["broad_confidence"] = [res["scores"][0] for res in results]
    return batch


def classify_broad(data, categories, on, classifier, batch_size=16):
    print("Converting to dataset...")
    dataset = Dataset.from_pandas(data)
    print("Processing...")
    dataset = dataset.map(
        classify_broad_dataset,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"classifier": classifier, "categories": categories, "on": on},
    )
    print("Converting back to dataframe...")
    final_data = dataset.to_pandas()
    return final_data
