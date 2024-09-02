import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import procrustes
from scipy.signal import correlate
from scipy.spatial.distance import cosine

# Load the two npy files
results_all_1 = np.load('path/to/first/X3D.npy')
results_all_2 = np.load('path/to/second/X3D.npy')

def compare_poses_procrustes(pose1, pose2):
    """
    Compares two poses using Procrustes analysis.
    
    :param pose1: First pose, shape (17, 3)
    :param pose2: Second pose, shape (17, 3)
    :return: Disparity between the two poses after alignment
    """
    _, _, disparity = procrustes(pose1, pose2)
    return disparity

def calculate_joint_angles(pose):
    """
    Calculates joint angles for a given pose.
    
    :param pose: Pose data, shape (17, 3)
    :return: Array of joint angles
    """
    angles = []
    for i in range(1, len(pose)):
        v1 = pose[i] - pose[0]
        v2 = pose[i-1] - pose[0]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    return np.array(angles)

def compare_poses_angles(pose1, pose2):
    """
    Compares two poses using joint angles.
    
    :param pose1: First pose, shape (17, 3)
    :param pose2: Second pose, shape (17, 3)
    :return: Average angle difference between the two poses
    """
    angles1 = calculate_joint_angles(pose1)
    angles2 = calculate_joint_angles(pose2)
    angle_diff = np.mean(np.abs(angles1 - angles2))
    return angle_diff

def compare_pose_sequences(seq1, seq2):
    """
    Compares two sequences of poses using temporal correlation.
    
    :param seq1: First sequence of poses, shape (num_frames, 17, 3)
    :param seq2: Second sequence of poses, shape (num_frames, 17, 3)
    :return: Average lag between the two sequences in frames
    """
    corr = [correlate(seq1[:, i, :].flatten(), seq2[:, i, :].flatten(), mode='full') for i in range(seq1.shape[1])]
    corr = np.array(corr)
    lag = np.mean(np.argmax(corr, axis=1) - (len(seq1) - 1))
    return lag

def compare_poses_cosine(pose1, pose2):
    """
    Compares two poses using cosine similarity.
    
    :param pose1: First pose, shape (17, 3)
    :param pose2: Second pose, shape (17, 3)
    :return: Cosine similarity between the two poses
    """
    similarity = 1 - cosine(pose1.flatten(), pose2.flatten())
    return similarity

def visualize_pose(pose, ax, color='b'):
    """
    Visualizes a single pose in 3D.
    
    :param pose: Pose data, shape (17, 3)
    :param ax: Matplotlib 3D axis
    :param color: Color of the pose
    """
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color)
    for i in range(len(pose)):
        ax.text(pose[i, 0], pose[i, 1], pose[i, 2], str(i), color=color)

def visualize_pose_sequence(seq1, seq2, frame_indices):
    """
    Visualizes two sequences of poses in 3D.
    
    :param seq1: First sequence of poses, shape (num_frames, 17, 3)
    :param seq2: Second sequence of poses, shape (num_frames, 17, 3)
    :param frame_indices: List of frame indices to visualize
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for frame in frame_indices:
        visualize_pose(seq1[frame], ax, color='b')
        visualize_pose(seq2[frame], ax, color='r')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Assuming results_all_1 and results_all_2 are loaded and have shape (num_frames, 17, 3)
    frame1, frame2 = 0, 10  # Example frame indices
    start1, end1 = 0, 30  # Example sequence start and end for first sequence
    start2, end2 = 10, 40  # Example sequence start and end for second sequence

    disparity = compare_poses_procrustes(results_all_1[frame1], results_all_2[frame2])
    print(f"Procrustes disparity: {disparity}")

    angle_diff = compare_poses_angles(results_all_1[frame1], results_all_2[frame2])
    print(f"Average angle difference: {angle_diff}")

    lag = compare_pose_sequences(results_all_1[start1:end1], results_all_2[start2:end2])
    print(f"Average lag between sequences: {lag} frames")

    similarity = compare_poses_cosine(results_all_1[frame1], results_all_2[frame2])
    print(f"Cosine similarity: {similarity}")

    # Visualize the first 5 frames of the sequences
    visualize_pose_sequence(results_all_1, results_all_2, frame_indices=[0, 1, 2, 3, 4])