import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from emotion_comparison import map_fer_to_sentiment, map_mafw_to_sentiment
import seaborn as sns
import matplotlib.pyplot as plt

def get_video_id_from_filename(filename):
    """Extract video ID from FER analysis filename."""
    # Assuming format like video_0000_f47.csv
    return filename.split('_')[1]

def compute_average_fer_scores(df):
    """Compute average FER scores for the first detected face (box0)."""
    emotion_cols = ['angry0', 'disgust0', 'fear0', 'happy0', 'sad0', 'surprise0', 'neutral0']
    return {col: df[col].mean() for col in emotion_cols}

def analyze_fer_results(fer_dir):
    """Process all FER result files and return sentiment predictions."""
    predictions = {}
    
    for filename in os.listdir(fer_dir):
        if not filename.endswith('.csv'):
            continue
            
        video_id = get_video_id_from_filename(filename)
        df = pd.read_csv(os.path.join(fer_dir, filename))
        
        # Compute average scores and map to sentiment
        avg_scores = compute_average_fer_scores(df)
        sentiment = map_fer_to_sentiment(avg_scores)
        predictions[f"{video_id:05d}.mp4"] = sentiment
        
    return predictions

def get_ground_truth(labels_file):
    """Get ground truth sentiments from MAFW labels."""
    df = pd.read_csv(labels_file)
    return {
        row['clip']: map_mafw_to_sentiment(row['single_label'])
        for _, row in df.iterrows()
    }

def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Sentiment Analysis Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def main():
    # Paths
    fer_dir = 'results/fer_analysis'
    labels_file = 'data/labels/single-set.csv'
    
    # Get predictions and ground truth
    predictions = analyze_fer_results(fer_dir)
    ground_truth = get_ground_truth(labels_file)
    
    # Prepare lists for sklearn metrics
    y_true = []
    y_pred = []
    
    # Match predictions with ground truth
    for clip, true_sentiment in ground_truth.items():
        if clip in predictions:
            y_true.append(true_sentiment)
            y_pred.append(predictions[clip])
    
    # Calculate metrics
    labels = ['positive', 'negative', 'neutral', 'other']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, labels)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'clip': list(ground_truth.keys()),
        'ground_truth': [ground_truth[clip] for clip in ground_truth.keys()],
        'prediction': [predictions.get(clip, 'missing') for clip in ground_truth.keys()]
    })
    results_df.to_csv('results/sentiment_analysis_results.csv', index=False)

if __name__ == "__main__":
    main() 