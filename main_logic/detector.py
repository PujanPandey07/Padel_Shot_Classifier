from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='models/yolov8n.pt'):
        self.model = YOLO(model_path)

    def detection_of_objects(self, frame):
        results = self.model.track(
            frame, persist=True, verbose=False,)

        detections = {'players': [], 'ball': None}
        for result in results:
            cls = int(result.boxes.cls[0])
            conf = float(result.boxes.conf[0])
            coords = result.boxes.xyxy[0].tolist()
            track_id = int(result.boxes.id[0])

            if cls == 0 and conf > 0.5:      # person
                detections['players'].append({
                    'id': track_id,
                    'box': coords,
                    'confidence': conf
                })
            elif cls == 32 and conf > 0.3:   # sports ball
                detections['ball'] = {
                    'box': coords,
                    'confidence': conf
                }

        return detections
