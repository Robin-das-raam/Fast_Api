# import argparse
# from webcam_display import display_webcam_with_inference, display_cctv_with_inference, display_video_file_with_inference

# def main(input_source, source_path=None):
#     if input_source == 'webcam':
#         display_webcam_with_inference()
#     elif input_source == 'cctv':
#         if source_path is None:
#             print("Error: CCTV URL is required for CCTV feed.")
#         else:
#             display_cctv_with_inference(source_path)
#     elif input_source == 'video':
#         if source_path is None:
#             print("Error: Video file path is required for video file.")
#         else:
#             display_video_file_with_inference(source_path)
#     else:
#         print("Error: Invalid input source. Choose from 'webcam', 'cctv', or 'video'.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run action recognition inference on various input sources.')
#     parser.add_argument('--source', type=str, required=True, choices=['webcam', 'cctv', 'video'],
#                         help="Input source for the inference. Choose from 'webcam', 'cctv', or 'video'.")
#     parser.add_argument('--path', type=str, required=False,
#                         help="Path to the source. Required for 'cctv' and 'video' sources.")

#     args = parser.parse_args()
#     main(args.source, args.path)

# inference_pipeline.py
import argparse
import cv2
import numpy as np
from webcam_display import display_webcam_with_inference, display_cctv_with_inference, display_video_file_with_inference
from preprocess import preprocess_frame
from model_loader import load_model
from config import MAX_FRAMES, ACTION_LABELS

model = load_model('/home/robinpc/Desktop/FastApi_prac/action_recognition_proj/models_file/my_vivit_model (1).h5')

def predict_action(frames):
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions, axis=-1)
    return ACTION_LABELS[predicted_class[0]], predictions

def write_video_with_predictions_from_file(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {input_video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_buffer = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_frame(frame)
        frame_buffer.append(preprocessed_frame)

        if len(frame_buffer) > MAX_FRAMES:  # Maintain a sliding window of max_frames
            frame_buffer.pop(0)

        if len(frame_buffer) == MAX_FRAMES:  # Only predict when we have enough frames
            frames = np.array(frame_buffer)
            action, predictions = predict_action(frames)

            # Check if predictions have expected dimensions
            if predictions.shape[-1] != len(ACTION_LABELS):
                raise ValueError(f"Unexpected predictions shape: {predictions.shape}")

            predicted_class = np.argmax(predictions[0], axis=-1)
            predicted_action = ACTION_LABELS[int(predicted_class)]
            
            cv2.putText(frame, f'Action: {predicted_action}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def main(input_source, source_path=None, output_path=None):
    if input_source == 'webcam':
        display_webcam_with_inference()
    elif input_source == 'cctv':
        if source_path is None:
            print("Error: CCTV URL is required for CCTV feed.")
        else:
            display_cctv_with_inference(source_path)
    elif input_source == 'video':
        if source_path is None:
            print("Error: Video file path is required for video file.")
        elif output_path is None:
            print("Error: Output video file path is required to save the result.")
        else:
            write_video_with_predictions_from_file(source_path, output_path)
    else:
        print("Error: Invalid input source. Choose from 'webcam', 'cctv', or 'video'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run action recognition inference on various input sources.')
    parser.add_argument('--source', type=str, required=True, choices=['webcam', 'cctv', 'video'],
                        help="Input source for the inference. Choose from 'webcam', 'cctv', or 'video'.")
    parser.add_argument('--path', type=str, required=False,
                        help="Path to the source. Required for 'cctv' and 'video' sources.")
    parser.add_argument('--output', type=str, required=False,
                        help="Output path to save the result. Required for 'video' source.")

    args = parser.parse_args()
    main(args.source, args.path, args.output)
