import cv2


# Extract features from the video frames
def extract_features(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total_frames


# converting Frame number to readable time stamps
def frame_to_timestamp(frame_number, fps):
    total_seconds = frame_number/fps
    minutes = int(total_seconds//60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:05.2f}"
