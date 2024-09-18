import argparse
from webcam_display import display_webcam_with_inference, display_cctv_with_inference, display_video_file_with_inference

def main(input_source, source_path=None):
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
        else:
            display_video_file_with_inference(source_path)
    else:
        print("Error: Invalid input source. Choose from 'webcam', 'cctv', or 'video'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run action recognition inference on various input sources.')
    parser.add_argument('--source', type=str, required=True, choices=['webcam', 'cctv', 'video'],
                        help="Input source for the inference. Choose from 'webcam', 'cctv', or 'video'.")
    parser.add_argument('--path', type=str, required=False,
                        help="Path to the source. Required for 'cctv' and 'video' sources.")

    args = parser.parse_args()
    main(args.source, args.path)
