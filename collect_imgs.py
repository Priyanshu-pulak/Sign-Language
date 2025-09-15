import cv2
from pathlib import Path

DATA_DIR = Path('./Data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

number_of_classes = 26
dataset_size = 200

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = DATA_DIR / str(j)
    class_dir.mkdir(parents=True, exist_ok=True)

    print(f'Collecting data for class {j}')

    counter = 0
    ready = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if not ready:
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                ready = True
        else:
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            file_path = class_dir / f'{counter}.jpg'
            cv2.imwrite(str(file_path), frame)
            counter += 1
            if counter >= dataset_size:
                break

cap.release()
cv2.destroyAllWindows()
