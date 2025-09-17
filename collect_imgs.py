import cv2
from pathlib import Path
import tkinter as tk
import time as t

def get_frame_size(cap) -> tuple[int, int]:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_width, frame_height

def get_screen_center(cap) -> tuple[int, int]:
    frame_width, frame_height = get_frame_size(cap)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    x = (screen_width - frame_width) // 2
    y = (screen_height - frame_height) // 2
    return x, y

def capture_images(class_id : int, num_images : int = 20, interval : float = .8):
    output_dir = Path('Data') / str(class_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    x, y = get_screen_center(cap)
    cv2.namedWindow("video_frame")
    cv2.moveWindow("video_frame", x, y)

    captured_images = 0
    ready = False
    start = None
    last_frame = None

    while captured_images < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        display_frame = frame.copy()
        last_frame = display_frame.copy()
        key = cv2.waitKey(1) & 0xFF

        if not ready:
            cv2.putText(display_frame, f"Alphabet : {chr(ord('a') + class_id)} | press 's to start and 'q' to quit", (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)

            if key == ord('s'):
                ready = True
                start = t.time()
            elif key == ord('q'):
                print(f"Quit early for alphabet : {chr(ord('a') + class_id)}")
                break
        else:
            cv2.putText(display_frame, f"Alphabet : {chr(ord('a') + class_id)} | image {captured_images + 1} / {num_images}", (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)
            if(t.time() - start) >= interval:
                img_path = output_dir / f"{captured_images + 1:03d}.jpg"
                cv2.imwrite(str(img_path), frame)
                captured_images += 1
                start = t.time()

        cv2.imshow("video_frame", display_frame)

    cap.release()
    return last_frame

num_classes = 26

for i in range(num_classes):
    last_frame = capture_images(i, 100, .8)
    if i < num_classes - 1 and last_frame is not None:
        while True:
            frame = last_frame.copy()
            msg = f"Finished class '{chr(ord('a') + i)}'. Press SPACE to continue or 'q' to quit."
            cv2.putText(frame, msg, (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)
            cv2.imshow("video_frame", frame)

            key = cv2.waitKey(0) & 0xFF
            if key == 32:  # SPACE key
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                exit()
cv2.destroyAllWindows()