import os
import time
import cv2

from settings import CUR_DIR


def crop_training_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    cnt = 1
    while True:
        _, frame = cap.read()
        h, w = frame.shape[:2]
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # press s to save
            cv2.imwrite(os.path.join(CUR_DIR, 'training_data', f"image{cnt}.jpg"), frame)
            cnt += 1
        elif cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
        # time.sleep(0.02)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    crop_training_frames(video_file="/media/main/Data/Task/BurgerTracker/00000004656000001.mp4")
