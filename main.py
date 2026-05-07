import cv2
import json
from main_logic.detector import Detector
from main_logic.pose_estimator import posedetector
from main_logic.shot_classifier import ShotClassifier
from main_logic.features import extract_features, frame_to_timestamp


def main():
    # Setup
    cap = cv2.VideoCapture('input/input_sample_video.mp4')
    if not cap.isOpened():
        raise FileNotFoundError("Could not open input/input_sample_video.mp4")
    fps, width, height, total_frames = extract_features(cap)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/output_video.mp4',
                          fourcc, fps, (width, height))

    # Initialize components
    detector = Detector()
    pose_estimator = posedetector()
    classifier = ShotClassifier()

    shot_results = []
    frame_count = 0
    ball_seen = 0
    player_seen = 0

    print(f"Processing {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for performance
        if frame_count % 2 != 0:
            out.write(frame)
            continue

        # Progress update
        if frame_count % 300 == 0:
            print(
                f"Processed {frame_count}/{total_frames} frames | "
                f"ball seen: {ball_seen} | player seen: {player_seen}"
            )

        # Step 1 - Detect
        detections = detector.detection_of_objects(frame)
        if detections['ball'] is not None:
            ball_seen += 1
        if detections['players']:
            player_seen += 1

        # Step 2 - Check for shot moment
        is_shot = classifier.is_shot_moment(
            detections['ball'], frame.shape, detections['players']
        )

        # Step 3 - Classify shot if moment detected
        if is_shot and detections['players']:
            # Use player closest to the ball
            bx1, by1, bx2, by2 = detections['ball']['box']
            ball_cx = (bx1 + bx2) / 2
            ball_cy = (by1 + by2) / 2
            player = min(
                detections['players'],
                key=lambda p: ((ball_cx - (p['box'][0] + p['box'][2]) / 2) ** 2 +
                               (ball_cy - (p['box'][1] + p['box'][3]) / 2) ** 2) ** 0.5
            )
            landmarks = pose_estimator.get_landmarks(
                frame, player['box']
            )
            shot_type = classifier.classify(landmarks, frame.shape)
            timestamp = frame_to_timestamp(frame_count, fps)

            # Save result
            shot_results.append({
                "frame": frame_count,
                "timestamp": timestamp,
                "player_id": player['id'],
                "shot_type": shot_type
            })

            # Draw on frame
            cv2.putText(frame, f"Shot: {shot_type}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            print(f"Shot detected: {shot_type} at {timestamp}")

        out.write(frame)

    # Save JSON output
    with open('output/shot_results.json', 'w') as f:
        json.dump(shot_results, f, indent=4)

    print(f"\nDone! Detected {len(shot_results)} shots total")
    print("Output saved to output/")

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
