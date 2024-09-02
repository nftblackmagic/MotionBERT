from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

def combine_video_audio(video_path, audio_path, output_path):
    # Load video and audio clips
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Set the audio of the video clip
    final_clip = video.set_audio(audio)

    # Write the result to a file
    final_clip.write_videofile(output_path)

    # Close the clips
    video.close()
    audio.close()

# Example usage
if __name__ == "__main__":
    video_file = "output_video.mp4"
    audio_file = "temp_audio.wav"
    output_file = "combined_video.mp4"

    combine_video_audio(video_file, audio_file, output_file)
