import ntpath
import glob
import os
import json
import pandas as pd
# import time
import cv2

from settings import CUR_DIR


def crop_training_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    cnt = 1136
    while True:
        _, frame = cap.read()
        # h, w = frame.shape[:2]
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # press s to save
            cv2.imwrite(os.path.join(CUR_DIR, 'training_data', f"image{cnt}.jpg"), frame)
            cnt += 1
        elif cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
        # time.sleep(0.02)

    cap.release()
    cv2.destroyAllWindows()


def crop_frame(training_dir):
    files = glob.glob(os.path.join(training_dir, "*.jpg"))
    annotated_data = pd.read_csv("")
    new_training_dir = ""
    y_pane = 0.3
    for f_path in files:
        file_name = ntpath.basename(f_path)
        new_path = os.path.join(new_training_dir, file_name)
        image = cv2.imread(f_path)
        h, w = image.shape[:2]
        diff_y = int(y_pane * h)
        if h == 1440 and w == 2560:
            x_pane = 0.6
            diff_x = int(x_pane * w)
            cropped_frame = image[diff_y:h, diff_x:w]
            f_annotated_indices = annotated_data.loc[annotated_data["#filename"] == file_name].index.tolist()
            for f_index in f_annotated_indices:
                origin_info = json.loads(annotated_data.iloc[f_index]["region_shape_attributes"])
                origin_info["x"] -= diff_x
                origin_info["y"] -= diff_y
                annotated_data["region_shape_attributes"][f_index] = json.dumps(origin_info)
            cv2.imwrite(new_path, cropped_frame)
        if h == 1080 and w == 1920:
            x_pane = 0.55
            diff_x = int(x_pane * w)
            cropped_frame = image[diff_y:h, diff_x:w]
            f_annotated_indices = annotated_data.loc[annotated_data["#filename"] == file_name].index.tolist()
            for f_index in f_annotated_indices:
                origin_info = json.loads(annotated_data.iloc[f_index]["region_shape_attributes"])
                origin_info["x"] -= diff_x
                origin_info["y"] -= diff_y
                annotated_data["region_shape_attributes"][f_index] = json.dumps(origin_info)
            cv2.imwrite(new_path, cropped_frame)
    annotated_data.to_csv("new.csv")

    return


if __name__ == '__main__':
    crop_frame(training_dir="")
    # frame = cv2.imread("")
    # cv2.rectangle(frame, (403, 703), (529, 834), (0, 0, 255))
    # cv2.imshow("Burger", frame)
    # cv2.waitKey()
