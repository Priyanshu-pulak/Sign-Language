import os
import cv2

DATA_DIR = './Data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 200

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

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
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
            counter += 1
            if counter >= dataset_size:
                break

cap.release()
cv2.destroyAllWindows()
