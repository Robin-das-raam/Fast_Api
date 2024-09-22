# preprocess.py
import cv2
import numpy as np

def preprocess_frame(frame, resize=(224, 224), dtype=np.float16):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, resize).astype(dtype)
    return frame
