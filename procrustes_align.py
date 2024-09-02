from scipy.spatial import procrustes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Add this import

def compare_poses_procrustes(pose1, pose2):
    """
    Compares two poses using Procrustes analysis.
    
    :param pose1: First pose, shape (17, 3)
    :param pose2: Second pose, shape (17, 3)
    :return: Disparity between the two poses after alignment
    """
    mtx1, mtx2, disparity = procrustes(pose1, pose2)
    return mtx1, mtx2, disparity

def load_and_compare_poses(file_path, frame1, frame2):
    """
    Loads poses from a .npy file and compares them using Procrustes analysis.
    
    :param file_path: Path to the .npy file containing the poses
    :param frame1: Index of the first frame to compare
    :param frame2: Index of the second frame to compare
    :return: Disparity between the two poses after alignment
    """
    data = np.load(file_path)
    pose1, pose2 = data[frame1], data[frame2]
    return compare_poses_procrustes(pose1, pose2)

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

# Example usage
if __name__ == "__main__":
    file_path = '2d/X3D.npy'
    
    # Predefined frame indices
    frame1 = 0  # Example index for the first frame
    frame2 = 500  # Example index for the second frame
    
    pose1, pose2, disparity = load_and_compare_poses(file_path, frame1, frame2)
    print(f"Disparity between the poses: {disparity}")
    
    # angle_diff = compare_poses_angles(pose1, pose2)
    # print(f"Average angle difference: {angle_diff}")
    
    # Load original poses for plotting
    data = np.load(file_path)
    original_pose1, original_pose2 = data[frame1], data[frame2]
    
    output_file = 'pose_comparison.png'
    plot_poses(original_pose1, original_pose2, output_file)
    print(f"Plot saved to {output_file}")
    
    # output_file_angles = 'pose_joint_angles.png'
    # plot_joint_angles(original_pose1, original_pose2, output_file_angles)
    # print(f"Joint angles plot saved to {output_file_angles}")
