from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)

    def detection_of_objects(self, frame):
        results = self.model.track(
            frame, persist=True, verbose=False,)

        detections = {'players': [], 'ball': None}
        best_ball_conf = -1.0
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            box_count = len(result.boxes)
            for i in range(box_count):
                cls = int(result.boxes.cls[i])
                conf = float(result.boxes.conf[i])
                coords = result.boxes.xyxy[i].tolist()
                if result.boxes.id is None:
                    track_id = -1
                else:
                    track_id = int(result.boxes.id[i])

                if cls == 0 and conf > 0.5:      # person
                    detections['players'].append({
                        'id': track_id,
                        'box': coords,
                        'confidence': conf
                    })
                elif cls == 32 and conf > 0.15:   # sports ball
                    if conf > best_ball_conf:
                        best_ball_conf = conf
                        detections['ball'] = {
                            'box': coords,
                            'confidence': conf
                        }

                # Keep only top 4 players by confidence
                detections['players'] = sorted(
                    detections['players'],
                    key=lambda x: x['confidence'],
                    reverse=True
                )[:4]

        return detections
