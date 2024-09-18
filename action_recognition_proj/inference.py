import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# DATA
INPUT_SHAPE = (60, 224, 224, 3) 
INPUT_SHAPE_2D = (224, 224, 3)
MAX_FRAMES = 60
NUM_CLASSES = 2
PROJECTION_DIM = 128
MOBILENET_NUM_LAYER = 40

# Define custom layers
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

class MobileNetV3FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, num_frames, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames

        self.mobilenetv3 = tf.keras.applications.MobileNetV3Small(
            input_shape=INPUT_SHAPE_2D, include_top=False, weights='imagenet'
        )
        self.reduced_mobilenetv3_model = tf.keras.Model(
            inputs=self.mobilenetv3.input, outputs=self.mobilenetv3.layers[MOBILENET_NUM_LAYER].output
        )

        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(PROJECTION_DIM)

    def call(self, inputs):
        frames = tf.split(inputs, num_or_size_splits=self.num_frames, axis=1)
        frame_features = [self.dense(self.pooling(self.reduced_mobilenetv3_model(tf.squeeze(frame, axis=1)))) for frame in frames]

        features = tf.stack(frame_features, axis=1)
        return features

# Load the model
model = keras.models.load_model(
    '/home/robinpc/Desktop/FastApi_prac/action_recognition_proj/models_file/my_vivit_model (1).h5',
    custom_objects={
        'MobileNetV3FeatureExtractor': MobileNetV3FeatureExtractor,
        'PositionalEncoder': PositionalEncoder
    }
)

def load_video(path, max_frames=MAX_FRAMES, resize=(224, 224), dtype=np.float16):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {path}")
    frames = []
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame.astype(dtype))
            frame_count += 1
    finally:
        cap.release()
    return frames

def capture_from_webcam(max_frames=MAX_FRAMES, resize=(224, 224), dtype=np.float16):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Error opening webcam")
    frames = []
    try:
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame.astype(dtype))
            frame_count += 1
    finally:
        cap.release()
    return frames

def capture_from_cctv(url, max_frames=MAX_FRAMES, resize=(224, 224), dtype=np.float16):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise ValueError(f"Error opening CCTV stream {url}")
    frames = []
    try:
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame.astype(dtype))
            frame_count += 1
    finally:
        cap.release()
    return frames

def write_video_with_predictions_from_file(input_video_path, output_video_path, predictions, frame_count, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {input_video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    action_labels = {0: 'no_action', 1: 'note_giving'}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        segment_idx = frame_idx // max_frames
        predicted_class = np.argmax(predictions[segment_idx], axis=1)[0]
        predicted_action = action_labels[int(predicted_class)]
        
        cv2.putText(frame, f'Action: {predicted_action}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def write_video_with_predictions_from_live(input_type, output_video_path, predictions, frame_count, max_frames=MAX_FRAMES, source=None):
    if input_type == "webcam":
        cap = cv2.VideoCapture(0)
    elif input_type == "cctv" and source:
        cap = cv2.VideoCapture(source)
    else:
        raise ValueError("Invalid input type for live video source. Use 'webcam' or 'cctv' with a valid URL.")
    
    if not cap.isOpened():
        raise ValueError(f"Error opening {input_type} stream")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    action_labels = {0: 'no_action', 1: 'note_giving'}
    frame_idx = 0
    while frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        segment_idx = frame_idx // max_frames
        predicted_class = np.argmax(predictions[segment_idx], axis=1)[0]
        predicted_action = action_labels[int(predicted_class)]
        
        cv2.putText(frame, f'Action: {predicted_action}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def get_video_source(input_type, source=None):
    if input_type == "file":
        if not source:
            raise ValueError("Source must be provided for file input type.")
        return load_video(source)
    elif input_type == "webcam":
        return capture_from_webcam()
    elif input_type == "cctv":
        if not source:
            raise ValueError("Source must be provided for cctv input type.")
        return capture_from_cctv(source)
    else:
        raise ValueError("Invalid input type. Choose from 'file', 'webcam', or 'cctv'.")

# Example usage
input_type = "webcam"  # Change this to "file", "webcam", or "cctv" as needed
source = None  # File path or URL for CCTV

if input_type == "file":
    source = "/home/robinpc/Desktop/FastApi_prac/action_recognition_proj/models_file/output_part_0.mp4"  # Example file path
elif input_type == "cctv":
    source = "rtsp://username:password@your_cctv_ip_address:port/stream"  # Example RTSP URL

video_frames = get_video_source(input_type, source)
frame_count = len(video_frames)

# Segment the video into chunks for prediction
num_segments = frame_count // MAX_FRAMES + (1 if frame_count % MAX_FRAMES != 0 else 0)
segments = [video_frames[i * MAX_FRAMES:(i + 1) * MAX_FRAMES] for i in range(num_segments)]

# Prepare the data for prediction
input_data = np.zeros((num_segments, MAX_FRAMES, 224, 224, 3), dtype=np.float16)
for i, segment in enumerate(segments):
    for j, frame in enumerate(segment):
        input_data[i, j] = frame

# Get predictions
predictions = model.predict(input_data)

output_video_path = "output_video.mp4"  # Example output path
if input_type == "file":
    write_video_with_predictions_from_file(source, output_video_path, predictions, frame_count)
else:
    write_video_with_predictions_from_live(input_type, output_video_path, predictions, frame_count, source)
