import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
            input_shape=(224, 224, 3), include_top=False, weights='imagenet'
        )
        self.reduced_mobilenetv3_model = tf.keras.Model(
            inputs=self.mobilenetv3.input, outputs=self.mobilenetv3.layers[40].output
        )

        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(128)

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

# Define labels
action_labels = {0: 'no_action', 1: 'note_giving'}

def preprocess_frame(frame, resize=(224, 224), dtype=np.float16):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, resize).astype(dtype)
    return frame

def predict_action(frames):
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions, axis=-1)
    return action_labels[predicted_class[0]]

def display_webcam_with_inference():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_buffer = []
    max_frames = 60  # Maximum frames for the sliding window
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        preprocessed_frame = preprocess_frame(frame)
        frame_buffer.append(preprocessed_frame)

        if len(frame_buffer) > max_frames:  # Maintain a sliding window of max_frames
            frame_buffer.pop(0)

        if len(frame_buffer) == max_frames:  # Only predict when we have enough frames
            frames = np.array(frame_buffer)
            action = predict_action(frames)
            
            # Display the action label on the frame
            cv2.putText(frame, f'Action: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Webcam Live Feed', frame)

        # Press 'q' to exit the webcam window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the webcam display with inference function
display_webcam_with_inference()
