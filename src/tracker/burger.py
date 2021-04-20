import time
import numpy as np
import cv2

from src.detector.burger import BurgerDetector
from src.tracker.nms import non_max_suppression_slow
from settings import IP_CAM_ADDRESS, VIDEO_PATH


class BurgerTracker:
    def __init__(self):
        self.burger_detector = BurgerDetector()
        self.tracking_ret = None
        self.burgers = []
        self.burger_attributes = []

    @staticmethod
    def detect_cheese_burger_color(roi_frame, cheese_burger=False):
        r_height, r_width = roi_frame.shape[:2]
        image = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        if cheese_burger:
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask0 = cv2.inRange(image, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(image, lower_red, upper_red)

            # join my masks
            mask = mask0 + mask1
        else:
            lower = np.array([22, 50, 0], dtype="uint8")
            upper = np.array([45, 255, 255], dtype="uint8")
            mask = cv2.inRange(image, lower, upper)
        # cv2.imshow("Cheese", mask)
        # cv2.waitKey()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        cheese_area = 0
        for cnt in contours:
            cheese_area += cv2.contourArea(cnt)
        if cheese_area > 0.1 * r_height * r_width:
            return True
        else:
            return False

    def run(self, video_file=VIDEO_PATH):
        if video_file == "":
            cap = cv2.VideoCapture(IP_CAM_ADDRESS)
        else:
            cap = cv2.VideoCapture(video_file)
        pane_x = 0.55
        pane_y = 0.3
        # detect_ret = False
        # detect_cnt = 0
        # pick_ret = False
        # pick_cnt = 0
        # end_ret = False
        # end_cnt = 0
        while cap.isOpened():
            _, frame = cap.read()
            height, width = frame.shape[:2]
            sub_frame = frame[int(pane_y * height):, int(pane_x * width):]
            # cv2.line(frame, (0, int(0.3 * height)), (width, int(0.3 * height)), (0, 0, 255), 5)
            # cv2.line(frame, (int(0.55 * width), 0), (int(0.55 * width), height), (0, 0, 255), 5)
            n_burgers, burger_classes, _ = self.burger_detector.detect_burger(frame=sub_frame)
            filter_ids = non_max_suppression_slow(boxes=np.array(n_burgers), keys=range(len(n_burgers)))
            for idx in filter_ids:
                n_burgers.pop(idx)
                burger_classes.pop(idx)
            # detect_ret = True
            # if len(n_burgers) != 0:
            #     print(f"[INFO] New detected: {len(n_burgers)}")
            for new_burger, new_burger_class in zip(n_burgers, burger_classes):
                if new_burger_class == "init":
                    left, top, right, bottom = new_burger
                    ret = False
                    for i, pre_burger in enumerate(self.burgers):
                        p_left, p_top, p_right, p_bottom = pre_burger
                        overlapped = non_max_suppression_slow(
                            boxes=np.array([[left, top, right, bottom], [p_left, p_top, p_right, p_bottom]]),
                            keys=[0, 1])
                        if overlapped:
                            ret = True
                            self.burger_attributes[i]["Detect_Count"] += 1
                            break
                    if not ret:
                        self.burgers.append(new_burger)
                        self.burger_attributes.append({"Start_Time": time.time(), "Detect_Count": 0})
            # if len(self.burgers) != 0:
            #     print(len(self.burgers))
            # if detect_ret and len(burgers) == 0:
            #     detect_cnt += 1
            #     if detect_cnt > 100:
            #         if len(self.burgers) < 5:
            #             self.burgers = []
            #         else:
            #             print("[INFO] Detecting Burger Finished")
            #             detect_ret = False
            #             detect_cnt = 0
            #             self.tracking_ret = "Detect"
            # if self.tracking_ret == "Detect":
            if burger_classes == ["pick"]:
                pi_left, pi_top, pi_right, pi_bottom = n_burgers[0]
                pi_x = 0.5 * (pi_right + pi_left)
                pi_y = pi_top + 0.8 * (pi_bottom - pi_top)
                for i, pre_burger in enumerate(self.burgers):
                    pre_left, pre_top, pre_right, pre_bottom = pre_burger
                    if pre_left <= pi_x <= pre_right and pre_top <= pi_y <= pre_bottom:
                        if time.time() - self.burger_attributes[i]["Start_Time"] < 50:
                            continue
                        self.burger_attributes[i].update({"Pick_Time": time.time()})
                        # pick_ret = True
            for i, pre_burger_attr in enumerate(self.burger_attributes):
                if i < len(self.burger_attributes) - 1:
                    if abs(self.burger_attributes[i + 1]["Start_Time"] - pre_burger_attr["Start_Time"]) < 20 and \
                            "Pick_Time" not in list(pre_burger_attr.keys()) and \
                            "Pick_Time" in list(self.burger_attributes[i + 1].keys()):
                        self.burger_attributes[i].update({"Pick_Time": self.burger_attributes[i + 1]["Pick_Time"] - 1})
                else:
                    if abs(self.burger_attributes[i - 1]["Start_Time"] - pre_burger_attr["Start_Time"]) < 20 and \
                            "Pick_Time" not in list(pre_burger_attr.keys()) and \
                            "Pick_Time" in list(self.burger_attributes[i - 1].keys()):
                        self.burger_attributes[i].update({"Pick_Time": self.burger_attributes[i - 1]["Pick_Time"] + 1})

            # if pick_ret and len(burgers) != 0:
            #     pick_cnt = 0
            # if pick_ret and len(burgers) == 0:
            #     pick_cnt += 1
            #     if pick_cnt > 500:
            #         for i, burger_attr in enumerate(self.burger_attributes):
            #             if "Pick_Time" not in list(burger_attr.keys()):
            #                 if i < len(self.burger_attributes) - 1:
            #                     self.burger_attributes[i].update(
            #                         {"Pick_Time": 0.5 * (self.burger_attributes[i - 1]["Pick_Time"] +
            #                                              self.burger_attributes[i + 1]["Pick_Time"])})
            #                 else:
            #                     self.burger_attributes[i].update(
            #                         {"Pick_Time": self.burger_attributes[i - 1]["Pick_Time"]})
            #         print("[INFO] Picking Burger Finished")
            #         pick_ret = False
            #         pick_cnt = 0
            #         self.tracking_ret = "Pick"
            # if self.tracking_ret == "Pick":
            pop_idx = []
            for i, pre_burger in enumerate(self.burgers):
                pre_left, pre_top, pre_right, pre_bottom = pre_burger
                if "Pick_Time" in list(self.burger_attributes[i].keys()):
                    if self.detect_cheese_burger_color(roi_frame=sub_frame[pre_top:pre_bottom, pre_left:pre_right]):
                        if "End_Time" not in list(self.burger_attributes[i].keys()) and \
                                time.time() - self.burger_attributes[i]["Pick_Time"] > 60:
                            self.burger_attributes[i].update({"End_Time": time.time()})
            for i, pre_burger_attr in enumerate(self.burger_attributes):
                if i < len(self.burger_attributes) - 1:
                    if abs(self.burger_attributes[i + 1]["Start_Time"] - pre_burger_attr["Start_Time"]) < 20 and \
                            "End_Time" not in list(pre_burger_attr.keys()) and \
                            "End_Time" in list(self.burger_attributes[i + 1].keys()):
                        self.burger_attributes[i].update({"End_Time": self.burger_attributes[i + 1]["End_Time"] - 1})
                else:
                    if abs(self.burger_attributes[i - 1]["Start_Time"] - pre_burger_attr["Start_Time"]) < 20 and \
                            "End_Time" not in list(pre_burger_attr.keys()) and \
                            "End_Time" in list(self.burger_attributes[i - 1].keys()):
                        self.burger_attributes[i].update({"End_Time": self.burger_attributes[i - 1]["End_Time"] + 1})

            for i, s_burger in enumerate(self.burgers):
                if "End_Time" in list(self.burger_attributes[i].keys()):
                    if time.time() - self.burger_attributes[i]["End_Time"] > 20:
                        if i not in pop_idx:
                            pop_idx.append(i)
                if time.time() - self.burger_attributes[i]["Start_Time"] > 200 and \
                        "Pick_Time" not in list(self.burger_attributes[i].keys()):
                    if i not in pop_idx:
                        pop_idx.append(i)
                if time.time() - self.burger_attributes[i]["Start_Time"] > 360:
                    if i not in pop_idx:
                        pop_idx.append(i)
                if time.time() - self.burger_attributes[i]["Start_Time"] > 5 and \
                        self.burger_attributes[i]["Detect_Count"] < 5:
                    if i not in pop_idx:
                        pop_idx.append(i)
                # s_left, s_top, s_right, s_bottom = s_burger
                # if not self.detect_cheese_burger_color(roi_frame=frame[s_top:s_bottom, s_left:s_right],
                #                                        cheese_burger=True):
                #     if i not in pop_idx:
                #         pop_idx.append(i)
                # if "End_Time" not in list(self.burger_attributes[i].keys()):
                #     end_ret = False
                # else:
                #     end_ret = True
            pop_idx = sorted(pop_idx, reverse=True)
            for p_idx in pop_idx:
                self.burger_attributes.pop(p_idx)
                self.burgers.pop(p_idx)
            for i, s_burger in enumerate(self.burgers):
                if self.burger_attributes[i]["Detect_Count"] >= 5:
                    if "Pick_Time" not in list(self.burger_attributes[i].keys()):
                        front_time = time.time() - self.burger_attributes[i]["Start_Time"]
                        cv2.circle(frame, (int(0.5 * (s_burger[0] + s_burger[2])) + int(pane_x * width),
                                           int(0.5 * (s_burger[1] + s_burger[3]) + pane_y * height)),
                                   20, (151, 142, 255), -1)
                    else:
                        front_time = self.burger_attributes[i]["Pick_Time"] - self.burger_attributes[i]["Start_Time"]
                        cv2.circle(frame, (int(0.5 * (s_burger[0] + s_burger[2])) + int(pane_x * width),
                                           int(0.5 * (s_burger[1] + s_burger[3]) + pane_y * height)),
                                   20, (139, 245, 255), -1)
                        if "End_Time" not in list(self.burger_attributes[i].keys()):
                            back_time = time.time() - self.burger_attributes[i]["Pick_Time"]
                        else:
                            back_time = self.burger_attributes[i]["End_Time"] - self.burger_attributes[i]["Pick_Time"]
                        back_sec = int(back_time % 60)
                        back_min = int(back_time // 60)
                        cv2.putText(frame, f"{back_min}:{back_sec}",
                                    (int(0.5 * (s_burger[0] + s_burger[2])) + int(pane_x * width),
                                     int(0.5 * (s_burger[1] + s_burger[3])) - 7 + int(pane_y * height)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    front_sec = int(front_time % 60)
                    front_min = int(front_time // 60)
                    cv2.putText(frame, f"{front_min}:{front_sec}",
                                (int(0.5 * (s_burger[0] + s_burger[2]) + pane_x * width),
                                 int(0.5 * (s_burger[1] + s_burger[3]) - 35 + pane_y * height)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # if end_ret and self.tracking_ret == "Pick":
            #     end_cnt += 1
            #     if end_cnt > 500:
            #         print("[INFO] Cooking Burger Finished")
            #         self.tracking_ret = None
            #         self.burgers = []
            #         self.burger_attributes = []
            #         end_cnt = 0
            #         end_ret = False

            cv2.imshow("Hamburger", cv2.resize(frame, None, fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()

        return


if __name__ == '__main__':
    # BurgerTracker().detect_cheese(roi_frame=cv2.imread(""))
    res = BurgerTracker().detect_cheese_burger_color(roi_frame=cv2.imread(""), cheese_burger=True)
    print(res)
