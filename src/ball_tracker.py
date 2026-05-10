from src.utils import get_box_center, get_distance


class BallTracker:

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.prev_ball = None
        self.ball_history = []
        self.stationary_threshold = 5

    def is_in_court(self, box):
        cx, cy = get_box_center(box)
        margin = 20

        if cx < margin or cx > self.frame_width - margin:
            return False
        if cy < self.frame_height * 0.10:
            return False
        if cy > self.frame_height * 0.80:
            return False

        return True

    def get_ball_in_play(self, ball_candidates):
        if not ball_candidates:
            self.prev_ball = None
            return None

        valid_balls = [
            b for b in ball_candidates
            if self.is_in_court(b['box'])
        ]

        if not valid_balls:
            self.prev_ball = None
            return None

        if self.prev_ball is None:
            ball = min(
                valid_balls,
                key=lambda b: get_box_center(b['box'])[1]
            )
            self._update_history(ball)
            self.prev_ball = ball
            return ball

        prev_cx, prev_cy = get_box_center(
            self.prev_ball['box']
        )

        best_ball = None
        max_movement = self.stationary_threshold

        for candidate in valid_balls:
            cx, cy = get_box_center(candidate['box'])
            movement = get_distance(
                (cx, cy), (prev_cx, prev_cy)
            )
            if movement > max_movement:
                max_movement = movement
                best_ball = candidate

        if best_ball is None:
            best_ball = min(
                valid_balls,
                key=lambda b: get_box_center(b['box'])[1]
            )

        self._update_history(best_ball)
        self.prev_ball = best_ball
        return best_ball

    def _update_history(self, ball):
        cx, cy = get_box_center(ball['box'])
        self.ball_history.append((cx, cy))
        if len(self.ball_history) > 5:
            self.ball_history.pop(0)

    def get_ball_velocity(self):
        if len(self.ball_history) < 2:
            return 0
        x1, y1 = self.ball_history[-2]
        x2, y2 = self.ball_history[-1]
        return get_distance((x1, y1), (x2, y2))