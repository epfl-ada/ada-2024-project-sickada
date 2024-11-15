import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from collections import defaultdict
from transformers import BartTokenizer, BartForSequenceClassification
import torch
import os
import os.path as op

figures_path = op.join('data', 'figures')

def classify_video_education(df, category_col='categories', title_col='title', description_col='description', tags_col='tags'):
    """
    Classifies videos as educational or not based on their title, description, and tags using BART.
    
    Parameters:
    - df: pandas DataFrame containing video metadata.
    - category_col: Column name containing the true categories (default: 'categories').
    - title_col: Column name containing the video title (default: 'title').
    - description_col: Column name containing the video description (default: 'description').
    - tags_col: Column name containing the video tags (default: 'tags').
    
    Returns:
    - A DataFrame with an added 'is_educational' column where 1 means educational and 0 means non-educational.
    """
    
    # Load pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    model.eval()  # Set the model to evaluation mode
    
    def is_educational_video(text):
        """Classify a video as educational or not based on title, description, and tags."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        # If the predicted class is 1 (education), we consider it educational.
        return 1 if predicted_class == 1 else 0

    # Classify videos as educational or not
    df['combined_text'] = df[title_col] + " " + df[description_col] + " " + df[tags_col]
    df['is_educational'] = df['combined_text'].apply(is_educational_video)
    
    return df


def create_confusion_matrix(df, category_col, predicted_col):
    """
    Creates a confusion matrix for each category in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the 'categories' and predicted 'is_educational' columns.
    - category_col: Column name containing the true category labels (e.g., 'categories').
    - predicted_col: Column name containing the predicted labels (e.g., 'is_educational').
    
    Returns:
    - A dictionary of confusion matrices for each category.
    """
    confusion_matrices = defaultdict(lambda: np.zeros((2, 2)))  # Initialize with a 2x2 matrix
    
    for idx, row in df.iterrows():
        true_label = row[category_col]
        predicted_label = row[predicted_col]
        
        # Update the confusion matrix for each category
        confusion_matrices[true_label][1, predicted_label] += 1  # [True label][Predicted label]
        
    return confusion_matrices


def compute_f1_scores(confusion_matrices):
    """
    Computes F1 score for each category using the confusion matrix.
    
    Parameters:
    - confusion_matrices: A dictionary where keys are categories and values are confusion matrices.
    
    Returns:
    - A dictionary of F1 scores for each category.
    """
    f1_scores = {}
    
    for category, cm in confusion_matrices.items():
        # Calculate F1 score based on confusion matrix: F1 = 2 * (precision * recall) / (precision + recall)
        tp = cm[1, 1]  # True positives
        fp = cm[0, 1]  # False positives
        fn = cm[1, 0]  # False negatives
        if tp + fp + fn == 0:  # Prevent division by zero
            f1_scores[category] = 0.0
        else:
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_scores[category] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_scores


def highlight_categories_by_f1(f1_scores, threshold=0.5):
    """
    Highlights categories where the F1 score exceeds the given threshold.
    
    Parameters:
    - f1_scores: A dictionary of F1 scores for each category.
    - threshold: The F1 score threshold to highlight categories.
    
    Returns:
    - List of categories where the F1 score exceeds the threshold.
    """
    highlighted_categories = [category for category, f1 in f1_scores.items() if f1 > threshold]
    return highlighted_categories


def save_confusion_matrices(confusion_matrices, output_folder="confusion_matrices"):
    """
    Saves each confusion matrix as a file in the specified folder.
    
    Parameters:
    - confusion_matrices: A dictionary of confusion matrices for each category.
    - output_folder: The folder where confusion matrices will be saved (default: "confusion_matrices").
    
    Returns:
    - None
    """
    # Create folder if not existing
    output_folder = os.path.join(figures_path, output_folder) # goes into figures because it is a visualisation tool
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save each confusion matrix as a file
    for category, cm in confusion_matrices.items():
        file_path = os.path.join(output_folder, f"confusion_matrix_{category}.txt")
        np.savetxt(file_path, cm, fmt='%d', delimiter=' ', header=f"Confusion Matrix for {category}", comments='')

def generate_classification_report(df, category_col='categories', predicted_col='is_educational', f1_threshold=0.5, output_folder="confusion_matrices"):
    """
    Generates a classification report for the educational classification task, including
    confusion matrices and F1 scores for each category.
    
    Parameters:
    - df: pandas DataFrame with 'categories' and 'is_educational' columns.
    - category_col: The column name containing the true categories.
    - predicted_col: The column name containing the predicted labels.
    - f1_threshold: The threshold above which categories will be highlighted based on their F1 score.
    - output_folder: Folder to save confusion matrix files (default: "confusion_matrices").
    
    Returns:
    - classification report as a dictionary with confusion matrices, F1 scores, and highlighted categories.
    """
    # Step 1: Classify videos as educational
    df = classify_video_education(df, category_col=category_col, title_col='title', description_col='description', tags_col='tags')
    
    # Step 2: Create confusion matrix for each category
    confusion_matrices = create_confusion_matrix(df, category_col, predicted_col)
    
    # Step 3: Compute F1 scores for each category
    f1_scores = compute_f1_scores(confusion_matrices)
    
    # Step 4: Highlight categories based on F1 score threshold
    highlighted_categories = highlight_categories_by_f1(f1_scores, f1_threshold)
    
    # Step 5: Save confusion matrices to folder
    save_confusion_matrices(confusion_matrices, output_folder)
    
    # Prepare the classification report
    report = {
        'confusion_matrices': confusion_matrices,
        'f1_scores': f1_scores,
        'highlighted_categories': highlighted_categories
    }
    
    return report

