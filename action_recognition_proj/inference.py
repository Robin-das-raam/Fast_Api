# Import necessary modules and define custom layers/functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# DATA
DATASET_NAME = "organmnist3d"
BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (60, 224, 224, 3) 
INPUT_SHAPE_2D = (224, 224, 3)
MAX_FRAMES = 60
NUM_CLASSES = 2

# OPTIMIZER
LEARNING_RATE = 1e-5

# TRAINING
EPOCHS = 30
DATA_NUM = 200

# TUBELET EMBEDDING
PATCH_SIZE = (8,8,8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6 
PROJECTION_DIM = 128
NUM_HEADS = 12 
NUM_LAYERS = 12 

# MOBILENETV
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
    frames = []
    try:
        frame_count = 0
        while True:
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

def write_video_with_predictions(input_video_path, output_video_path, predictions, frame_count, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(input_video_path)
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

# Preprocess the video
video_path = '/home/robinpc/Desktop/FastApi_prac/action_recognition_proj/models_file/output_part_9.mp4'
video_frames = load_video(video_path)
frame_count = len(video_frames)

# Segment the video into chunks for prediction
num_segments = frame_count // MAX_FRAMES + (1 if frame_count % MAX_FRAMES != 0 else 0)
predictions = []

for i in range(num_segments):
    start_frame = i * MAX_FRAMES
    end_frame = min((i + 1) * MAX_FRAMES, frame_count)
    segment_frames = video_frames[start_frame:end_frame]

    # If the segment is shorter than MAX_FRAMES, pad with the last frame
    while len(segment_frames) < MAX_FRAMES:
        segment_frames.append(segment_frames[-1])
    
    segment_frames = np.expand_dims(segment_frames, axis=0)  # Add batch dimension
    segment_predictions = model.predict(segment_frames)
    predictions.append(segment_predictions)

# Write the output video with the predicted actions
output_video_path = '/home/robinpc/Desktop/FastApi_prac/action_recognition_proj/output_file/output_with_actions.mp4'
write_video_with_predictions(video_path, output_video_path, predictions, frame_count)
print(f"Output video saved to: {output_video_path}")
