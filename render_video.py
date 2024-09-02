import cv2
import numpy as np
from lib.utils.vismo import render_and_save_3d
from procrustes_align import load_and_compare_poses, plot_disparities
from seq_analyse import analyze_pose_movement, draw_movement_amplitude, analyze_bouncing_movement, draw_vertical_velocity
from moviepy.editor import VideoFileClip, AudioFileClip
import os


def downsample_data(data, target_length):
    """
    Downsamples the data to the target length.
    
    :param data: Original pose data
    :param target_length: Target number of frames
    :return: Downsampled pose data
    """
    indices = np.linspace(0, len(data) - 1, target_length).astype(int)
    return data[indices]

# Example usage
if __name__ == "__main__":
    file_path1 = 'output_data1/X3D.npy'
    file_path2 = 'output_data2/X3D.npy'
    
    # Extract FPS from video files
    video_path1 = 'data1.mp4'  # Replace with actual video file path
    video_path2 = 'data2.mp4'  # Replace with actual video file path
    
    output_path = './output_video.mp4'  # Replace with desired output file path
    
    cap1 = cv2.VideoCapture(video_path1)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    cap1.release()
    
    cap2 = cv2.VideoCapture(video_path2)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    cap2.release()
    
    print(f"FPS for video 1: {fps1}")
    print(f"FPS for video 2: {fps2}")
    
    data1 = np.load(file_path1)
    data2 = np.load(file_path2)
    
    # Calculate the ratio of FPS to determine downsampling
    fps_ratio = fps1 / fps2
    fps = fps1
    
    if fps_ratio > 1:
        # Downsample data1
        target_length = int(len(data1) / fps_ratio)
        data1 = downsample_data(data1, target_length)
        fps = fps2
    elif fps_ratio < 1:
        # Downsample data2
        target_length = int(len(data2) * fps_ratio)
        data2 = downsample_data(data2, target_length)
        fps = fps1
    
    # Ensure both datasets have the same number of frames
    min_frames = min(len(data1), len(data2))
    data1 = data1[:min_frames]
    data2 = data2[:min_frames]
    
    disparities, data1, data2 = load_and_compare_poses(file_path1, file_path2, fps1, fps2)
    print(f"Disparities between the poses: {disparities}")
    # Plot and save the disparities chart
    disparity_chart_file = 'disparity_chart.png'
    plot_disparities(disparities, disparity_chart_file)
    print(f"Disparity chart saved to {disparity_chart_file}")
    
    result1 = analyze_pose_movement(data1, fps=fps1, window_size=1)
    result2 = analyze_pose_movement(data2, fps=fps2, window_size=1)
    
    draw_movement_amplitude(
        [result1['movement_amplitude'], result2['movement_amplitude']],
        [fps1, fps2],
        ['Video 1', 'Video 2'],
        'movement_amplitude_comparison.png'
    )
    print("Movement amplitude comparison graph saved as 'movement_amplitude_comparison.png'")
    
    res1 = analyze_bouncing_movement(data1, fps=fps1)
    res2 = analyze_bouncing_movement(data2, fps=fps2)

    draw_vertical_velocity(
        [res1['vertical_velocity'], res2['vertical_velocity']],
        [fps1, fps2],
        ['Video 1', 'Video 2'],
        'vertical_velocity_comparison.png'
    )
    print("Vertical velocity comparison graph saved as 'vertical_velocity_comparison.png'")
    # Render the video
    render_and_save_3d(
        data1, 
        data2, 
        output_path, 
        fps=fps, 
        disparities=disparities, 
    )

    print(f"Video saved to {output_path}")
