
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
            if not ret or frame_count >= max_frames:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame.astype(dtype))
            frame_count += 1
    finally:
        cap.release()

    while len(frames) < max_frames:
        frames.append(frames[-1].astype(dtype))

    return np.array(frames)

# Preprocess the video
video_path = '/home/robinpc/Desktop/FastApi_prac/action_recognition_proj/models_file/output_part_0.mp4'
video_frames = load_video(video_path)

# Ensure the input shape matches the model's input shape
video_frames = np.expand_dims(video_frames, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(video_frames)
print("Predictions:", predictions)
