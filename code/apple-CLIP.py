import sys
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import face_recognition
from open_clip import create_model_from_pretrained, get_tokenizer

# Add the project root to the Python path
sys.path.append(os.path.join(os.getcwd(), 'code'))

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

politician_expressions = [
    "Pleased",
    "Happy",
    "Positive",
    "Angry",
    "Sad",
    "Negative",
    "Composed",
    "Neutral",
    "Formal"
]

positive_emotions = ["pleased", "happy", "positive"]
negative_emotions = ["angry", "sad", "negative"]
neutral_emotions = ["composed", "neutral", "formal"]

def detect_and_crop_face(image, padding=100):
    image_np = np.array(image)
    face_locations = face_recognition.face_locations(image_np, model="hog")
    
    if not face_locations:
        return None

    top, right, bottom, left = face_locations[0]
    
    # Apply padding
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(image_np.shape[0], bottom + padding)
    right = min(image_np.shape[1], right + padding)

    face_image = image_np[top:bottom, left:right]
    return Image.fromarray(face_image)

def classify_emotion(image, model, preprocess, tokenizer):
    face_image = detect_and_crop_face(image)
    if face_image is not None:
        image_input = preprocess(face_image).unsqueeze(0)
        text = tokenizer(politician_expressions, context_length=model.context_length)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Calculate similarity scores
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        emotion_probs = dict(zip([expr.lower() for expr in politician_expressions], similarity[0].tolist()))
        return emotion_probs, "Yes"
    else:
        return None, "No"

def process_batch(image_files, start_index, model, preprocess, tokenizer):
    columns = ['image_name', 'face', 'face.label', 'label', 'label.label']
    df = pd.DataFrame(columns=columns)

    for i, filename in enumerate(image_files, start=start_index):
        print(f"Processing image {i}/{total_images}: {filename}")
        
        image_path = os.path.join('fop', filename)
        image = Image.open(image_path).convert('RGB')

        emotion_probs, face = classify_emotion(image, model, preprocess, tokenizer)

        if emotion_probs:
            positive_prob = sum(emotion_probs[emotion] for emotion in positive_emotions)
            negative_prob = sum(emotion_probs[emotion] for emotion in negative_emotions)
            neutral_prob = sum(emotion_probs[emotion] for emotion in neutral_emotions)

            max_prob = max(positive_prob, negative_prob, neutral_prob)
            
            if positive_prob == max_prob:
                label = "Positive Emotion"
            elif negative_prob == max_prob:
                label = "Negative Emotion"
            else:
                label = "Neutral Expression"
              
            print(f"  Face detected. Label: {label}")
            print(f"  Emotion probabilities: {emotion_probs}")
        else:
            label = 'Neutral Expression'
            print("  No face detected. Label: Neutral Expression")

        df = df.append({
            'image_name': filename,
            'face': face,
            'face.label': '',
            'label': label,
            'label.label': ''
        }, ignore_index=True)

    return df

print("Loading AppleCLIP model...")
model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
tokenizer = get_tokenizer('ViT-H-14')

# Get all image files
all_images = [f for f in os.listdir('fop') if f.endswith('.jpg')]
all_images.sort()  # Ensure consistent ordering
total_images = len(all_images)

# Determine the batch to process
batch_size = 100
progress_file = 'apple_clip_progress.txt'

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        start_index = int(f.read().strip())
else:
    start_index = 0

end_index = min(start_index + batch_size, total_images)

print(f"Processing images {start_index + 1} to {end_index} out of {total_images}")

# Process the current batch
current_batch = all_images[start_index:end_index]
df = process_batch(current_batch, start_index + 1, model, preprocess, tokenizer)

# Save the results
output_file = f'results/appleCLIP-emotion_{start_index + 1}-{end_index}.csv'
df.to_csv(output_file, index=False, quoting=1)

print(f"\nBatch processing complete. Results have been saved to {output_file}")

# Update progress
with open(progress_file, 'w') as f:
    f.write(str(end_index))

# Combine all CSV files if this was the last batch
if end_index == total_images:
    print("All batches processed. Combining CSV files...")
    all_df = pd.DataFrame()
    for i in range(0, total_images, batch_size):
        batch_file = f'results/appleCLIP-emotion_{i + 1}-{min(i + batch_size, total_images)}.csv'
        if os.path.exists(batch_file):
            batch_df = pd.read_csv(batch_file)
            all_df = pd.concat([all_df, batch_df], ignore_index=True)
            os.remove(batch_file)  # Delete the intermediary file
    
    final_output = 'results/appleCLIP.csv'
    all_df.to_csv(final_output, index=False, quoting=1)
    print(f"All results combined and saved to {final_output}")
    print("Intermediary files have been deleted.")
else:
    print(f"Run the script again to process the next batch. Progress: {end_index}/{total_images}")