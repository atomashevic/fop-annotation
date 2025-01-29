import sys
import os
import pandas as pd
import torch

# Add the project root to the Python path
sys.path.append(os.path.join(os.getcwd(), 'code'))

from PIL import Image
from detector import Detector

print("Initializing Detector...")
detector = Detector(show_type='weak', problem_id='states')

columns = ['image_name', 'face', 'face.label', 'label', 'label.label']
df = pd.DataFrame(columns=columns)

print("Starting image processing...")

# Loop through all images in the 'fop' folder
total_images = len([f for f in os.listdir('fop') if f.endswith('.jpg')])
processed_images = 0

def classify_emotion(emotions):
    positive_prob = emotions.get('positive', 0)
    negative_prob = emotions.get('negative', 0)
    neutral_prob = emotions.get('neutral', 0)

    max_prob = max(positive_prob, negative_prob, neutral_prob)
    
    if max_prob == positive_prob:
        return 'Positive Emotion'
    elif max_prob == negative_prob:
        return 'Negative Emotion'
    else:
        return 'Neutral Expression'

for filename in os.listdir('fop'):
    if filename.endswith('.jpg'):
        processed_images += 1
        print(f"Processing image {processed_images}/{total_images}: {filename}")
        
        image_path = os.path.join('fop', filename)
        image = Image.open(image_path)

        # Perform state detection
        detected_states = detector.detect_emotions(image)

        if detected_states:
            # Get the emotions for the first (and usually only) detected face
            emotions = detected_states[0]['emotions']
            
            # Classify the emotion using our new function
            label = classify_emotion(emotions)

            face = 'Yes'
            print(f"  Face detected. Label: {label}")
            print(f"  Emotion probabilities: {emotions}")
        else:
            # If no face detected or all scores are zero
            label = 'Neutral Expression'
            face = 'No'
            print("  No face detected or all scores are zero. Label: Neutral Expression")

        # Append the results to the DataFrame
        df = df.append({
            'image_name': filename,
            'face': face,
            'face.label': '',
            'label': label,
            'label.label': ''
        }, ignore_index=True)

# Save the DataFrame to a CSV file
output_file = 'results/CNN-strong.csv'
df.to_csv(output_file, index=False, quoting=1)  # quoting=1 ensures all fields are quoted

print(f"\nProcessing complete. Results have been saved to {output_file}")
print(f"Total images processed: {processed_images}")