import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, 'utils', 'model')

THRESHOLD = 0.5
INPUT_MEAN = 127.5
INPUT_STD = 127.5
OVERLAP_THRESH = 0.2

TPU = True
LOCAL = False

IP_CAM_ADDRESS = "rtsp://admin:1234567m@62.149.95.22:555"
VIDEO_PATH = ""
