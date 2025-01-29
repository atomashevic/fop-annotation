import os
import sys
import pandas as pd
import torch
from PIL import Image
import face_recognition
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# Add the project root to the Python path
sys.path.append(os.path.join(os.getcwd(), 'code'))

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def classify_emotion(image, tokenizer, model, image_processor, context_len):
    face_image = detect_and_crop_face(image)
    if face_image is None:
        return None, "No"

    prompt = ("Analyze the facial expression in this image. Classify it as one of three categories: "
              "Positive emotion expression (Pleased, Happy, Excited ), Negative emotion expression (Angry, Sad, Frustrated), Neutral expression (Composed, Reserved, Formal). Respond only with the emotion category: Positive, Negative, Neutral.")

    image_tensor = image_processor.preprocess(face_image, return_tensors='pt')['pixel_values'].half().cuda()

    input_ids = tokenizer(prompt).input_ids
    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=32,
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return output.lower(), "Yes"

def process_batch(image_files, start_index, tokenizer, model, image_processor, context_len):
    columns = ['image_name', 'face', 'face.label', 'label', 'label.label']
    df = pd.DataFrame(columns=columns)

    for i, filename in enumerate(image_files, start=start_index):
        print(f"Processing image {i}/{total_images}: {filename}")
        
        image_path = os.path.join('fop', filename)
        image = Image.open(image_path).convert('RGB')

        emotion, face = classify_emotion(image, tokenizer, model, image_processor, context_len)

        if emotion:
            if emotion in ["pleased", "happy", "positive"]:
                label = "Positive Emotion"
            elif emotion in ["angry", "sad", "negative"]:
                label = "Negative Emotion"
            else:
                label = "Neutral Expression"
            
            print(f"  Face detected. Label: {label}")
            print(f"  Detected emotion: {emotion}")
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

print("Loading LLaVA model...")
model_path = "liuhaotian/llava-v1.5-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# Get all image files
all_images = [f for f in os.listdir('fop') if f.endswith('.jpg')]
all_images.sort()  # Ensure consistent ordering
total_images = len(all_images)

# Determine the batch to process
batch_size = 100
progress_file = 'llava_progress.txt'

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        start_index = int(f.read().strip())
else:
    start_index = 0

end_index = min(start_index + batch_size, total_images)

print(f"Processing images {start_index + 1} to {end_index} out of {total_images}")

# Process the current batch
current_batch = all_images[start_index:end_index]
df = process_batch(current_batch, start_index + 1, tokenizer, model, image_processor, context_len)

# Save the results
output_file = f'results/LLaVA-emotion_{start_index + 1}-{end_index}.csv'
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
        batch_file = f'results/LLaVA-emotion_{i + 1}-{min(i + batch_size, total_images)}.csv'
        if os.path.exists(batch_file):
            batch_df = pd.read_csv(batch_file)
            all_df = pd.concat([all_df, batch_df], ignore_index=True)
            os.remove(batch_file)  # Delete the intermediary file
    
    final_output = 'results/LLaVA.csv'
    all_df.to_csv(final_output, index=False, quoting=1)
    print(f"All results combined and saved to {final_output}")
    print("Intermediary files have been deleted.")
else:
    print(f"Run the script again to process the next batch. Progress: {end_index}/{total_images}")