from ultralytics import YOLO
import cv2


class PoseEstimator:

    def __init__(self,
                 model_path='models/yolov8s-pose.pt'):
        self.model = YOLO(model_path)

    def get_keypoints_full_frame(self, frame,
                                  player_detections):
        """
        Run pose on each player crop individually.
        This works better for far/small players because
        we zoom into each player before pose detection.
        """
        persons = []
        h, w = frame.shape[:2]

        for player in player_detections:
            x1, y1, x2, y2 = [int(c) for c in player['box']]

            # Add padding around player
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # Crop player region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Upscale small crops for better detection
            crop_h, crop_w = crop.shape[:2]
            if crop_w < 100 or crop_h < 100:
                scale = max(100/crop_w, 100/crop_h)
                crop = cv2.resize(
                    crop,
                    (int(crop_w*scale), int(crop_h*scale))
                )

            # Run pose on crop
            results = self.model(
                crop,
                verbose=False,
                conf=0.1
            )

            if not results or \
               results[0].keypoints is None:
                continue

            kps_data = results[0].keypoints.xy\
                .cpu().numpy()

            if len(kps_data) == 0:
                continue

            # Use first detected person
            kps_xy = kps_data[0]
            crop_h2, crop_w2 = crop.shape[:2]

            # Normalize by crop dimensions
            normalized = []
            for kp in kps_xy:
                normalized.append(kp[0] / crop_w2)
                normalized.append(kp[1] / crop_h2)

            # Skip if mostly zero keypoints
            non_zero = sum(
                1 for i in range(0, len(normalized), 2)
                if normalized[i] > 0 or normalized[i+1] > 0
            )
            if non_zero < 6:
                continue

            persons.append({
                'id': player['id'],
                'box': player['box'],
                'confidence': player['confidence'],
                'keypoints': normalized
            })

        return persons