import cv2

def check_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

video_path = "./123.mp4"
fps = check_video_fps(video_path)
print(f"The video FPS is: {fps}")