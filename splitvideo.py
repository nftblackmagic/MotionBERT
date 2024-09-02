import cv2
import os
import argparse

def extract_frames(video_path, output_dir, fps=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    print("video_fps: ", video_fps)
    
    # Use video's FPS if not specified
    if fps is None:
        fps = video_fps
    
    # Calculate frame interval
    interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {saved_count} frames at {fps} fps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video_path", help="Path to the input video file")
    parser.add_argument("--output_dir", help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, help="Frames per second to extract (default: video's FPS)")

    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir, args.fps)
