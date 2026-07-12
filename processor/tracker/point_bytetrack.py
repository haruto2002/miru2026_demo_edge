from dataclasses import dataclass

import cv2
import numpy as np

from processor.modules.tracker.point_bytetrack.byte_tracker import PointBYTETracker


@dataclass
class PointByteTrackArgs:
    track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 10.0
    distance_metric: str = "euclidean"
    frame_rate: int = 30


class PointByteTrackTracker:
    def __init__(
        self,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        distance_metric: str,
        frame_rate: int,
        point_size: int = 3,
    ):
        args = PointByteTrackArgs(
            track_thresh, track_buffer, match_thresh, distance_metric
        )
        self.tracker = PointBYTETracker(args, frame_rate=frame_rate)
        self.point_size = point_size

    def update(self, dets: np.ndarray):
        """
        dets: [[x, y, score], ...]
        """
        return "point", self.tracker.update(dets)

    def draw(self, image, tracks):
        for i, track in enumerate(tracks):
            x, y = track.point
            obj_id = track.track_id
            color = self.get_color(abs(obj_id))
            cv2.circle(image, (int(x), int(y)), self.point_size, color, -1)

        return image

    def get_color(self, label_id):
        color = (
            int((37 * label_id) % 255),
            int((17 * label_id) % 255),
            int((29 * label_id) % 255),
        )
        return color
