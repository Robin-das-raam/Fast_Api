# custom_layers.py
import tensorflow as tf
from tensorflow.keras import layers
from config import MOBILENET_NUM_LAYER, PROJECTION_DIM

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
            input_shape=(224, 224, 3), include_top=False, weights='imagenet'
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
