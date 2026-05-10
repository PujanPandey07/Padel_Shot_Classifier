import cv2
import numpy as np


def get_video_properties(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total_frames


def frame_to_timestamp(frame_number, fps):
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def get_box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5


def detect_shot_optical_flow(prev_frame, curr_frame,
                              player_box):
    """
    Detect shot using optical flow in player region.
    High average flow magnitude = fast arm movement = shot
    """
    if prev_frame is None or curr_frame is None:
        return 0.0

    x1, y1, x2, y2 = [int(c) for c in player_box]

    # Safety checks
    h, w = prev_frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    prev_crop = prev_frame[y1:y2, x1:x2]
    curr_crop = curr_frame[y1:y2, x1:x2]

    if prev_crop.size == 0 or curr_crop.size == 0:
        return 0.0

    if prev_crop.shape != curr_crop.shape:
        return 0.0

    prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)

    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(
            flow[..., 0], flow[..., 1]
        )
        return float(np.mean(magnitude))
    except:
        return 0.0


def is_in_main_court(box, frame_width, frame_height):
    """
    Filter players to main court region only.
    Ignores players on side courts and background.
    Adjust these values based on your video layout.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    in_x = frame_width * 0.05 < cx < frame_width * 0.95
    in_y = frame_height * 0.08 < cy < frame_height * 0.95

    return in_x and in_y