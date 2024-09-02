import numpy as np
from scipy.signal import find_peaks
import cv2
import matplotlib.pyplot as plt

def analyze_bouncing_movement(keypoints_sequence, fps=30, window_size=1):
    """
    Analyze the movement amplitude of a 3D human pose sequence.
    
    :param keypoints_sequence: numpy array of shape (frames, 17, 3) containing 3D coordinates of 17 keypoints
    :param fps: frames per second of the sequence
    :param window_size: size of the sliding window in seconds
    :return: dictionary containing movement analysis results
    """
    
    # Calculate the center of mass (COM) for each frame using only leg, trunk, and head joints
    leg_trunk_head_indices = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13]  # Indices for relevant joints
    com_sequence = np.mean(keypoints_sequence[:, leg_trunk_head_indices, :], axis=1)
    
    # Calculate the velocity of the COM
    velocity = np.diff(com_sequence, axis=0)
    # Detect bouncing using Y-axis movement of COM
    vertical_velocity = velocity[:, 1]  # Y-axis is vertical
    
    return {
        'vertical_velocity': vertical_velocity
    }

def analyze_pose_movement(keypoints_sequence, fps=30, window_size=1):
    """
    Analyze the movement amplitude of a 3D human pose sequence.
    
    :param keypoints_sequence: numpy array of shape (frames, 17, 3) containing 3D coordinates of 17 keypoints
    :param fps: frames per second of the sequence
    :param window_size: size of the sliding window in seconds
    :return: dictionary containing movement analysis results
    """
    
    # Calculate the center of mass (COM) for each frame
    com_sequence = np.mean(keypoints_sequence, axis=1)
    
    # Calculate the velocity of the COM
    velocity = np.diff(com_sequence, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    
    # Calculate the acceleration of the COM
    acceleration = np.diff(velocity, axis=0)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    
    # Calculate movement amplitude using a sliding window
    window_frames = int(window_size * fps)
    movement_amplitude = np.convolve(speed, np.ones(window_frames), 'valid') / window_frames
    
    # Classify movement intensity
    soft_threshold = np.percentile(movement_amplitude, 25)
    heavy_threshold = np.percentile(movement_amplitude, 75)
    
    movement_intensity = np.zeros_like(movement_amplitude, dtype=str)
    movement_intensity[movement_amplitude <= soft_threshold] = 'soft'
    movement_intensity[(movement_amplitude > soft_threshold) & (movement_amplitude <= heavy_threshold)] = 'medium'
    movement_intensity[movement_amplitude > heavy_threshold] = 'heavy'

    
    return {
        'movement_amplitude': movement_amplitude,
        'movement_intensity': movement_intensity,
        'com_sequence': com_sequence,
        'speed': speed,
        'acceleration_magnitude': acceleration_magnitude
    }

def print_analysis_results(results, fps=30):
    """
    Print a summary of the movement analysis results.
    
    :param results: dictionary containing movement analysis results
    :param fps: frames per second of the sequence
    """
    print("Movement Analysis Results:")
    print(f"Sequence duration: {len(results['movement_amplitude']) / fps:.2f} seconds")
    print(f"Average movement amplitude: {np.mean(results['movement_amplitude']):.4f}")
    print(f"Max movement amplitude: {np.max(results['movement_amplitude']):.4f}")
    
    intensity_counts = np.unique(results['movement_intensity'], return_counts=True)
    for intensity, count in zip(*intensity_counts):
        print(f"{intensity.capitalize()} movement: {count / len(results['movement_intensity']) * 100:.2f}% of the time")
    
    print(f"Number of detected bounces: {len(results['bouncing_frames'])}")
    if len(results['bouncing_frames']) > 0:
        print(f"Bouncing detected at frames: {results['bouncing_frames']}")
        print(f"Bouncing timestamps: {results['bouncing_frames'] / fps}")

def draw_movement_amplitude(movement_amplitudes, fps_list, labels, output_path):
    """
    Draw multiple movement amplitude graphs on the same axis and save it as an image.
    
    :param movement_amplitudes: list of numpy arrays of movement amplitude values
    :param fps_list: list of frames per second for each sequence
    :param labels: list of labels for each amplitude curve
    :param output_path: path to save the output image
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red'] + [np.random.rand(3,) for _ in range(len(movement_amplitudes)-2)]
    
    for i, (movement_amplitude, fps) in enumerate(zip(movement_amplitudes, fps_list)):
        time = np.arange(len(movement_amplitude)) / fps
        plt.plot(time, movement_amplitude, label=labels[i], color=colors[i])
    
    plt.title('Movement Amplitude Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Movement Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def draw_vertical_velocity(vertical_velocities, fps_list, labels, output_path):
    """
    Draw multiple vertical velocity graphs on the same axis and save it as an image.
    
    :param vertical_velocities: list of numpy arrays of vertical velocity values
    :param fps_list: list of frames per second for each sequence
    :param labels: list of labels for each velocity curve
    :param output_path: path to save the output image
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red'] + [np.random.rand(3,) for _ in range(len(vertical_velocities)-2)]
    
    for i, (vertical_velocity, fps) in enumerate(zip(vertical_velocities, fps_list)):
        time = np.arange(len(vertical_velocity)) / fps
        plt.plot(time, vertical_velocity, label=labels[i], color=colors[i])
    
    plt.title('Vertical Velocity Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Vertical Velocity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Generate some dummy data for demonstration
    file_path1 = 'output_data1/X3D.npy'
    file_path2 = 'output_data2/X3D.npy'
    
    # Extract FPS from video files
    video_path1 = 'data1.mp4'  # Replace with actual video file path
    video_path2 = 'data2.mp4'  # Replace with actual video file path
    
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
    
    print(data1.shape)
    
    result = analyze_pose_movement(data1, fps=fps1, window_size=1)
    print_analysis_results(result, fps=fps1)
    
    # Draw and save the movement amplitude graph
    draw_movement_amplitude(result['movement_amplitude'], fps1, 'movement_amplitude.png')
    print("Movement amplitude graph saved as 'movement_amplitude.png'")
    
    result = analyze_pose_movement(data2, fps=fps2, window_size=1)
    print_analysis_results(result, fps=fps2)
    
    # Draw and save the movement amplitude graph
    draw_movement_amplitude(result['movement_amplitude'], fps2, 'movement_amplitude2.png')
    print("Movement amplitude graph saved as 'movement_amplitude2.png'")
    
    res = analyze_bouncing_movement(data1, fps=fps1)
    draw_vertical_velocity([res['vertical_velocity']], [fps1], ['Video 1'], 'vertical_velocity.png') 
    print("Vertical velocity graph saved as 'vertical_velocity.png'")
    
    res = analyze_bouncing_movement(data2, fps=fps2)
    draw_vertical_velocity([res['vertical_velocity']], [fps2], ['Video 2'], 'vertical_velocity2.png')
    print("Vertical velocity graph saved as 'vertical_velocity2.png'")
    
