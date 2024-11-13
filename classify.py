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

def classify_dataset(batch, candidate_labels, on, classifier, multi_label=False) :
    texts = batch[on]
    results = classifier(texts, candidate_labels, multi_label=multi_label)

    labels_per_text = []
    for result in results:
        scores, labels = result['scores'], result['labels']
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

def classify(data, candidate_labels, classifier, on='text', multi_label=True, batch_size=16):
    print('Convering to dataset...')
    dataset = Dataset.from_pandas(data)
    print('Processing...')
    dataset = dataset.map(
        classify_dataset, 
        batched=True, 
        batch_size=batch_size, 
        fn_kwargs={'classifier': classifier, 
                'candidate_labels': candidate_labels, 
                'on': on, 
                'multi_label': multi_label},
    )
    print('Converting back to DataFrame...')
    final_data = dataset.to_pandas()
    return final_data
    

