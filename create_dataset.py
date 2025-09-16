from pathlib import Path
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

DATA_DIR = Path('./Data')

data = []
labels = []

for dir_path in DATA_DIR.iterdir():
    for img_path in dir_path.iterdir():

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = []

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)

            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_path.name)
            else:
                print(f"Incomplete landmarks in image : {img_path}, skipping...")
        else:
            print(f"No hand landmarks detected in image: {img_path}")

output_file = Path("data.pickle")
with output_file.open('wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
