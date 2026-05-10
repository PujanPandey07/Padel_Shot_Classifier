from ultralytics import YOLO


class Detector:

    def __init__(self, model_path='models/yolov8s.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            tracker='bytetrack.yaml'
        )

        detections = {
            'players': [],
            'balls': []
        }

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                track_id = int(box.id[0]) \
                    if box.id is not None else -1

                if cls == 0 and conf > 0.6:
                    detections['players'].append({
                        'id': track_id,
                        'box': coords,
                        'confidence': conf
                    })
                elif cls == 32 and conf > 0.3:
                    detections['balls'].append({
                        'box': coords,
                        'confidence': conf
                    })

        detections['players'] = sorted(
            detections['players'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:4]

        return detections