import cv2
import json
import csv
from src.detector import Detector
from src.pose_estimator import PoseEstimator
from src.ball_tracker import BallTracker
from src.shot_classifier import ShotClassifier
from src.utils import (get_video_properties,
                       frame_to_timestamp,
                       detect_shot_optical_flow,
                       )


def main():
    cap = cv2.VideoCapture('input/infernce_sample_video.mp4')
    fps, width, height, total_frames = \
        get_video_properties(cap)

    scale = 0.5
    out_w = int(width * scale)
    out_h = int(height * scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        'output/output_video.mp4',
        fourcc, fps, (out_w, out_h)
    )

    detector    = Detector()
    pose_est    = PoseEstimator()
    ball_tracker = BallTracker(width, height)
    classifier  = ShotClassifier()

    shot_results = []
    shot_display = {}
    frame_count  = 0
    prev_frame   = None

    print(f"Processing {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 3 != 0:
            frame_small = cv2.resize(frame, (out_w, out_h))
            out.write(frame_small)
            prev_frame = frame.copy()
            continue

        if frame_count % 300 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames}"
                  f" ({pct:.1f}%)")

        # Step 1 - Detect players and balls
        detections = detector.detect(frame)

        # Step 2 - Get ball in play
        ball = ball_tracker.get_ball_in_play(
            detections['balls']
        )

        

        persons = pose_est.get_keypoints_full_frame(
    frame, detections['players']
)
        
    


        # Step 5 - Draw all detected players
        for player in detections['players']:
            slot_id = classifier.get_nearest_slot(
                player['box']
            )
            x1, y1, x2, y2 = [
                int(c) for c in player['box']
            ]
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (255, 0, 0), 2
            )
            cv2.putText(
                frame, f"P{slot_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2
            )

        # Step 6 - Process each person for shot detection
        for person in persons:
            slot_id = classifier.get_nearest_slot(
                person['box']
            )
            keypoints = person['keypoints']

            # Calculate optical flow for this player
            flow_mag = 0.0
            if prev_frame is not None:
                flow_mag = detect_shot_optical_flow(
                    prev_frame, frame, person['box']
                )

            is_shot = classifier.is_shot_moment(
                slot_id, keypoints,
                ball, person['box'],
                flow_magnitude=flow_mag
            )

            if is_shot:
                shot_type = classifier.classify(keypoints)
                timestamp = frame_to_timestamp(
                    frame_count, fps
                )

                shot_results.append({
                    "frame": frame_count,
                    "timestamp": timestamp,
                    "player_id": slot_id,
                    "shot_type": shot_type
                })

                shot_display[slot_id] = {
                    'type': shot_type,
                    'frames_left': 40
                }

                print(f"Shot! P{slot_id}: "
                      f"{shot_type} at {timestamp}")

            # Show shot label
            if slot_id in shot_display and \
               shot_display[slot_id]['frames_left'] > 0:

                x1, y1, x2, y2 = [
                    int(c) for c in person['box']
                ]
                label = shot_display[slot_id]['type']

                color = (0, 255, 0)
                if label == 'Backhand':
                    color = (0, 165, 255)
                elif label == 'Smash':
                    color = (0, 0, 255)

                cv2.putText(
                    frame, f"{label.upper()}!",
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 3
                )
                shot_display[slot_id]['frames_left'] -= 1

        # Draw ball
        if ball:
            bx1, by1, bx2, by2 = [
                int(c) for c in ball['box']
            ]
            cv2.rectangle(
                frame,
                (bx1, by1), (bx2, by2),
                (0, 255, 0), 2
            )
            cv2.putText(
                frame, "Ball",
                (bx1, by1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1
            )

        prev_frame = frame.copy()
        frame_small = cv2.resize(frame, (out_w, out_h))
        out.write(frame_small)

    # Save JSON
    with open('output/shot_results.json', 'w') as f:
        json.dump(shot_results, f, indent=4)

    # Save CSV
    with open('output/shot_results.csv', 'w',
               newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'timestamp',
            'player_id', 'shot_type'
        ])
        writer.writeheader()
        writer.writerows(shot_results)

    print(f"\nDone! {len(shot_results)} shots detected")
    cap.release()
    out.release()


if __name__ == "__main__":
    main()