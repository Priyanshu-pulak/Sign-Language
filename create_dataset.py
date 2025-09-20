from pathlib import Path
import pickle
import mediapipe as mp
import cv2
import math
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

DATA_DIR = Path('./Data')

data = []
labels = []

def rotate_landmarks(data_aux, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotated = []

    for i in range(0, len(data_aux), 2):
        x, y = data_aux[i], data_aux[i+1]
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        rotated.extend([x_new, y_new])
    return rotated

for dir_path in DATA_DIR.iterdir():
    if dir_path.is_dir():
        for img_path in dir_path.iterdir():
            if not img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load image: {img_path}, skipping...")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                data_aux = []

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - wrist_x)
                    data_aux.append(lm.y - wrist_y)

                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_path.name)
                    
                    for _ in range(2):
                        angle = random.uniform(-15, 15)
                        rotated = rotate_landmarks(data_aux, angle)
                        data.append(rotated)
                        labels.append(dir_path.name)
                else:
                    print(f"Incomplete landmarks in image : {img_path}, skipping...")
            else:
                print(f"No hand landmarks detected in image: {img_path}")

output_file = Path("data.pkl")
with output_file.open('wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
