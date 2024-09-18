# model_loader.py
import tensorflow as tf
from tensorflow import keras
from custom_layers import MobileNetV3FeatureExtractor, PositionalEncoder

def load_model(model_path):
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'MobileNetV3FeatureExtractor': MobileNetV3FeatureExtractor,
            'PositionalEncoder': PositionalEncoder
        }
    )
    return model
