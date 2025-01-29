import sys
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from PIL import Image
import face_recognition

# Add the project root to the Python path
sys.path.append(os.path.join(os.getcwd(), 'code'))

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

politician_expressions = [
    "Pleased",
    "Happy",
    "Excited",
    # "Positive",
    "Angry",
    "Sad",
    "Frustrated",
    # "Negative",
    "Composed",
    "Reserved",
    # "Neutral",
    "Formal"
]

positive_emotions = ["pleased", "happy", "excited"]
negative_emotions = ["angry", "sad", "frustrated"]
neutral_emotions = ["composed", "reserved", "formal"]

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

def classify_emotion(image, processor, model):
    text_inputs = processor(text=politician_expressions, return_tensors='pt', padding=True)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)

    face_image = detect_and_crop_face(image)
    if face_image is not None:
        image_inputs = processor(images=face_image, return_tensors='pt')
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * model.logit_scale.exp()
        probs = logits_per_image.softmax(dim=1).squeeze(0).tolist()
        
        # Make sure the keys in emotion_probs match exactly with politician_expressions
        emotion_probs = dict(zip([expr.lower() for expr in politician_expressions], probs))
        
        return emotion_probs, "Yes"
    else:
        return None, "No"

def process_batch(image_files, start_index, model, preprocess):
    columns = ['image_name', 'face', 'face.label', 'label', 'label.label']
    df = pd.DataFrame(columns=columns)

    for i, filename in enumerate(image_files, start=start_index):
        print(f"Processing image {i}/{total_images}: {filename}")
        
        image_path = os.path.join('fop', filename)
        image = Image.open(image_path).convert('RGB')

        emotion_probs, face = classify_emotion(image, model, preprocess)

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
    
print("Loading CLIP model...")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")

# Get all image files
all_images = [f for f in os.listdir('fop') if f.endswith('.jpg')]
all_images.sort()  # Ensure consistent ordering
total_images = len(all_images)

# Determine the batch to process
batch_size = 100
progress_file = 'clip_ne_progress.txt'

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        start_index = int(f.read().strip())
else:
    start_index = 0

end_index = min(start_index + batch_size, total_images)

print(f"Processing images {start_index + 1} to {end_index} out of {total_images}")

# Process the current batch
current_batch = all_images[start_index:end_index]
df = process_batch(current_batch, start_index + 1, processor, model)

# Save the results
output_file = f'results/CLIP-NE-emotion_{start_index + 1}-{end_index}.csv'
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
        batch_file = f'results/CLIP-NE-emotion_{i + 1}-{min(i + batch_size, total_images)}.csv'
        if os.path.exists(batch_file):
            batch_df = pd.read_csv(batch_file)
            all_df = pd.concat([all_df, batch_df], ignore_index=True)
            os.remove(batch_file)  # Delete the intermediary file
    
    final_output = 'results/CLIP-NE.csv'
    all_df.to_csv(final_output, index=False, quoting=1)
    print(f"All results combined and saved to {final_output}")
    print("Intermediary files have been deleted.")
else:
    print(f"Run the script again to process the next batch. Progress: {end_index}/{total_images}")