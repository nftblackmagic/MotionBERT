from scipy.spatial import procrustes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Add this import
import cv2

def compare_poses_procrustes(pose1, pose2):
    """
    Compares two poses using Procrustes analysis.
    
    :param pose1: First pose, shape (17, 3)
    :param pose2: Second pose, shape (17, 3)
    :return: Disparity between the two poses after alignment
    """
    mtx1, mtx2, disparity = procrustes(pose1, pose2)
    return mtx1, mtx2, disparity

def downsample_data(data, target_length):
    """
    Downsamples the data to the target length.
    
    :param data: Original pose data
    :param target_length: Target number of frames
    :return: Downsampled pose data
    """
    indices = np.linspace(0, len(data) - 1, target_length).astype(int)
    return data[indices]

def plot_disparities(disparities, output_file):
    """
    Plots the disparities over time and saves the chart to a file.
    
    :param disparities: List of disparity values
    :param output_file: Path to save the chart image
    """
    plt.figure(figsize=(10, 6))
    plt.plot(disparities)
    plt.title('Pose Disparities Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Disparity')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def load_and_compare_poses(file_path1, file_path2, fps1, fps2):
    """
    Loads poses from two .npy files and compares them using Procrustes analysis.
    
    :param file_path1: Path to the first .npy file containing the poses
    :param file_path2: Path to the second .npy file containing the poses
    :param fps1: Frames per second of the first file
    :param fps2: Frames per second of the second file
    :return: List of disparities between the poses after alignment
    """
    data1 = np.load(file_path1)
    data2 = np.load(file_path2)
    
    # Calculate the ratio of FPS to determine downsampling
    fps_ratio = fps1 / fps2
    
    if fps_ratio > 1:
        # Downsample data1
        target_length = int(len(data1) / fps_ratio)
        data1 = downsample_data(data1, target_length)
    elif fps_ratio < 1:
        # Downsample data2
        target_length = int(len(data2) * fps_ratio)
        data2 = downsample_data(data2, target_length)
    
    # Ensure both datasets have the same number of frames
    min_frames = min(len(data1), len(data2))
    data1 = data1[:min_frames]
    data2 = data2[:min_frames]
    
    disparities = []
    
    for i in range(min_frames):
        pose1, pose2 = data1[i], data2[i]
        _, _, disparity = compare_poses_procrustes(pose1, pose2)
        disparities.append(disparity)
    
    return disparities, data1, data2

def plot_poses(pose1, pose2, output_file):
    """
    Plots the original poses and saves the plot to a file.
    
    :param pose1: Original first pose
    :param pose2: Original second pose
    :param output_file: Path to save the plot image
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original poses
    ax.scatter(pose1[:, 0], pose1[:, 1], pose1[:, 2], c='r', label='Original Pose 1')
    ax.scatter(pose2[:, 0], pose2[:, 1], pose2[:, 2], c='b', label='Original Pose 2')
    
    ax.legend()
    plt.savefig(output_file)
    plt.close()

def combine_videos_with_disparity(data1, data2, disparities, output_path, fps):
    """
    Creates a video from pose data arrays and adds live disparity scores.
    
    :param data1: Pose data for the first video
    :param data2: Pose data for the second video
    :param disparities: List of disparity scores
    :param output_path: Path to save the combined video
    :param fps: Frames per second for the output video
    """
    height, width = 480, 640  # You can adjust these values as needed
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

    for frame_count in range(len(data1)):
        # Create blank frames
        frame1 = np.zeros((height, width, 3), dtype=np.uint8)
        frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw pose keypoints on frames
        for point in data1[frame_count]:
            x, y = int(point[0] * width), int(point[1] * height)
            cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)

        for point in data2[frame_count]:
            x, y = int(point[0] * width), int(point[1] * height)
            cv2.circle(frame2, (x, y), 5, (0, 255, 0), -1)

        # Combine frames side by side
        combined_frame = np.hstack((frame1, frame2))

        # Add disparity score as text overlay
        disparity_text = f"Disparity: {disparities[frame_count]:.4f}"
        cv2.putText(combined_frame, disparity_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(combined_frame)

    out.release()

# Example usage
if __name__ == "__main__":
    file_path1 = '2d/X3D.npy'
    file_path2 = '2d1/X3D1.npy'
    
    # Extract FPS from video files
    video_path1 = '2d/X3D.mp4'  # Replace with actual video file path
    video_path2 = '2d1/X3D1.mp4'  # Replace with actual video file path
    
    cap1 = cv2.VideoCapture(video_path1)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    cap1.release()
    
    cap2 = cv2.VideoCapture(video_path2)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    cap2.release()
    
    print(f"FPS for video 1: {fps1}")
    print(f"FPS for video 2: {fps2}")
    
    disparities, data1, data2 = load_and_compare_poses(file_path1, file_path2, fps1, fps2)
    print(f"Disparities between the poses: {disparities}")
    # Plot and save the disparities chart
    disparity_chart_file = 'disparity_chart.png'
    plot_disparities(disparities, disparity_chart_file)
    print(f"Disparity chart saved to {disparity_chart_file}")

    # # Create video from pose data and add disparity scores
    # output_video_path = 'combined_video_with_disparity.mp4'
    # combine_videos_with_disparity(data1, data2, disparities, output_video_path, fps1)
    # print(f"Combined video saved to {output_video_path}")
    
    # # Load original poses for plotting
    # data1 = np.load(file_path1)
    # data2 = np.load(file_path2)
    
    # # Plot the first frame as an example
    # original_pose1, original_pose2 = data1[0], data2[0]
    # output_file = 'pose_comparison.png'
    # plot_poses(original_pose1, original_pose2, output_file)
    # print(f"Plot saved to {output_file}")
