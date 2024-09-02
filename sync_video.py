import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import librosa


def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_name = f"temp_audio_{video_path.split('/')[-1].split('.')[0]}.wav"
    audio.write_audiofile(audio_name)
    video.close()
    return audio_name

def find_sync_point(audio1, audio2, fps1, fps2):
    rate1, data1 = wavfile.read(audio1)
    rate2, data2 = wavfile.read(audio2)
    
    if rate1 != rate2:
        raise ValueError("Audio sample rates do not match")
    
    y1, sr1 = librosa.load(audio1, sr=None)
    y2, sr2 = librosa.load(audio2, sr=None)
    
    # Extract chroma features
    chroma1 = librosa.feature.chroma_stft(y=y1, sr=sr1)
    chroma2 = librosa.feature.chroma_stft(y=y2, sr=sr2)
    
    # Compute the alignment path using DTW
    D, wp = librosa.sequence.dtw(X=chroma1, Y=chroma2)
    
    # Calculate the offset
    offset = wp[-1, 0] - wp[-1, 1]
    print(f"Offset between the two audio files is: {offset} frames")    
    
    # Ensure data is mono
    if len(data1.shape) > 1:
        data1 = np.mean(data1, axis=1)
    if len(data2.shape) > 1:
        data2 = np.mean(data2, axis=1)
    
    correlation = correlate(data1.astype(float), data2.astype(float), mode='full')
    print(correlation.shape, np.argmax(correlation), len(data1), rate1)
    sync_point = np.argmax(correlation) - len(data1) + 1
    
    
    if sync_point >= 0:
        start1, start2 = 0, sync_point
    else:
        start1, start2 = -sync_point, 0
    
    common_length = min(len(data1) - start1, len(data2) - start2)
    end1 = start1 + common_length
    end2 = start2 + common_length
    
    # Calculate start and end frames
    start_frame1, end_frame1 = int(start1 * fps1) / rate1, int(end1 * fps1) / rate1
    start_frame2, end_frame2 = int(start2 * fps2) / rate2, int(end2 * fps2) / rate2

    return start_frame1, end_frame1, start_frame2, end_frame2

def sync_and_cut_videos(video1_path, video2_path, output_dir):
    # Extract audio from both videos
    audio1_path = extract_audio(video1_path)
    audio2_path = extract_audio(video2_path)
    
    # Load video clips to get fps
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)
    
    # Find the sync point
    start_frame1, end_frame1, start_frame2, end_frame2 = find_sync_point(audio1_path, audio2_path, video1.fps, video2.fps)
    
    # Synchronize videos using frame information
    # print(start_frame1, end_frame1, start_frame2, end_frame2)
    video1 = video1.subclip(start_frame1 / video1.fps, end_frame1 / video1.fps)
    video2 = video2.subclip(start_frame2 / video2.fps, end_frame2 / video2.fps)
    
    print(start_frame1 / video1.fps, end_frame1 / video1.fps, start_frame2 / video2.fps, end_frame2 / video2.fps, video1.duration, video2.duration)
    
    # # Cut to the same length using frame information
    # common_duration = min(video1.duration, video2.duration)
    # end_frame1 = start_frame1 + int(common_duration * video1.fps)
    # end_frame2 = start_frame2 + int(common_duration * video2.fps)
    
    # video1 = video1.subclip(start_frame1 / video1.fps, end_frame1 / video1.fps)
    # video2 = video2.subclip(start_frame2 / video2.fps, end_frame2 / video2.fps)
    
    # Create new videos based on original video settings
    video1_output = video1.copy()
    video2_output = video2.copy()
    
    # After synchronizing the videos
    print(f"Original video1 duration: {video1.duration}")
    print(f"Original video2 duration: {video2.duration}")
    print(f"Synchronized video1 duration: {video1_output.duration}")
    print(f"Synchronized video2 duration: {video2_output.duration}")
    
    # Write the new videos
    output_dir = os.path.dirname(output_dir)  # Or specify a different output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        video1_output.write_videofile(os.path.join(output_dir, f"synced_{os.path.basename(video1_path)}"))
    except Exception as e:
        print(f"Error writing video1: {str(e)}")
        print(f"Video1 duration: {video1_output.duration}")
        print(f"Video1 audio duration: {video1_output.audio.duration if video1_output.audio else 'No audio'}")

    try:
        video2_output.write_videofile(os.path.join(output_dir, f"synced_{os.path.basename(video2_path)}"))
    except Exception as e:
        print(f"Error writing video2: {str(e)}")
        print(f"Video2 duration: {video2_output.duration}")
        print(f"Video2 audio duration: {video2_output.audio.duration if video2_output.audio else 'No audio'}")
    
    # Close video files
    video1.close()
    video2.close()
    video1_output.close()
    video2_output.close()

# Example usage
video1_path = "./1235.mp4"
video2_path = "./1246.mp4"
output_dir = "./output"
sync_and_cut_videos(video1_path, video2_path, output_dir)