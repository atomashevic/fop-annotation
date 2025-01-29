import os
import random
import numpy as np
from PIL import Image
from detector import Detector

def process_images(input_folder='fop', output_folder='debug_images', num_images=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the Detector
    detector = Detector(show_type='weak', problem_id='states')

    all_images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    for filename in selected_images:
        print(f"Processing image: {filename}")
        
        image_path = os.path.join(input_folder, filename)
        
        # Save original image
        original_image = Image.open(image_path)
        original_output_path = os.path.join(output_folder, f"original_{filename}")
        original_image.save(original_output_path)
        print(f"Saved original image: {original_output_path}")

        # Detect faces using the Detector
        detected_faces = detector.detect_emotions(original_image)

        if detected_faces:
            for i, face_data in enumerate(detected_faces):
                # Convert box coordinates to the format expected by PIL
                left, top, width, height = face_data['box']
                right, bottom = left + width, top + height
                
                try:
                    face_image = original_image.crop((left, top, right, bottom))
                    face_output_path = os.path.join(output_folder, f"face_{i+1}_{filename}")
                    face_image.save(face_output_path)
                    print(f"Saved face {i+1}: {face_output_path}")

                    # Print emotion probabilities
                    print(f"Emotion probabilities for face {i+1}:")
                    for emotion, prob in face_data['emotions'].items():
                        print(f"  {emotion}: {prob:.3f}")
                except ValueError as e:
                    print(f"Error cropping face {i+1}: {e}")
                    print(f"Box coordinates: {face_data['box']}")
        else:
            print(f"No faces detected in {filename}")

if __name__ == "__main__":
    process_images()
    print("Processing complete. Check the 'debug_images' folder for results.")