# webcam_display.py
import cv2
import numpy as np
from preprocess import preprocess_frame
from config import MAX_FRAMES, ACTION_LABELS
from model_loader import load_model

model = load_model('/home/robinpc/Desktop/FastApi_prac/action_recognition_proj/models_file/my_vivit_model (1).h5')

def predict_action(frames):
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions, axis=-1)
    return ACTION_LABELS[predicted_class[0]]

def display_feed_with_inference(cap):
    frame_buffer = []
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        preprocessed_frame = preprocess_frame(frame)
        frame_buffer.append(preprocessed_frame)

        if len(frame_buffer) > MAX_FRAMES:  # Maintain a sliding window of max_frames
            frame_buffer.pop(0)

        if len(frame_buffer) == MAX_FRAMES:  # Only predict when we have enough frames
            frames = np.array(frame_buffer)
            action = predict_action(frames)
            
            # Display the action label on the frame
            cv2.putText(frame, f'Action: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Live Feed', frame)

        # Press 'q' to exit the feed window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def display_webcam_with_inference():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    display_feed_with_inference(cap)

def display_cctv_with_inference(cctv_url):
    cap = cv2.VideoCapture(cctv_url)
    
    if not cap.isOpened():
        print(f"Error: Could not open CCTV feed at {cctv_url}.")
        return
    
    display_feed_with_inference(cap)

def display_video_file_with_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}.")
        return
    
    display_feed_with_inference(cap)

if __name__ == "__main__":

    # Uncomment one of the following lines to test the respective functionality
    display_webcam_with_inference()
    # display_cctv_with_inference('http://your_cctv_url')
    # display_video_file_with_inference('/path/to/your/video_file.mp4')
