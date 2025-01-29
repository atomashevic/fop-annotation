import face_recognition
import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

EMOTIONS_MAP = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

STATES_MAP = {
    0: 'positive',
    1: 'negative',
    2: 'neutral',
}

device = "cuda" if torch.cuda.is_available() else "cpu"

class Detector():
    def __init__(self, show_type='strong', problem_id='emotions'):

        model = models.vgg13()
        if problem_id == 'emotions':
            model.classifier[6] = nn.Linear(in_features=4096, out_features=len(EMOTIONS_MAP), bias=True)
            self.output_map = EMOTIONS_MAP
        elif problem_id == 'states':
            model.classifier[6] = nn.Linear(in_features=4096, out_features=len(STATES_MAP), bias=True)
            self.output_map = STATES_MAP
        
        model_path = f'models/{show_type}_{problem_id}.pkl'
        model_weights = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_weights['model_state'])
        self.model = model.to(device)
        self.model.eval()
    
    def detect_emotions(self, image, model="hog"):
        image = np.array(image)
        face_locations = face_recognition.face_locations(image, model=model)
        detected_emotions = []
        for face_location in face_locations:
            top, right, bottom, left = face_location

            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image = pil_image.resize((224, 224))
            # pil_image.show()

            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_image = transform(pil_image).to(device)

            output = self.model(input_image.unsqueeze(0))
            output = F.softmax(output, dim=1)

            emotions = {}
            for i, emotion in enumerate(output[0]):
                emotions[self.output_map[i]] = round(emotion.item(), 3)

            detected_emotions.append({
                'box': [left, top, right-left, bottom-top],
                'emotions': emotions
            })
        return detected_emotions

    def top_emotion(self, image):
        detected_emotions = self.detect_emotions(image)
        output = []
        for person in detected_emotions:
            emotions = person['emotions']
            max_value = max(emotions.values())
            max_key = max(emotions, key=emotions.get)
            output.append({
                'box': person['box'],
                'top_emotion': (max_key, max_value)
            })
        return output