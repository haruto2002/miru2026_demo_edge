from dataclasses import dataclass

import cv2
import numpy as np

from processor.modules.tracker.bytetrack.byte_tracker import BYTETracker


@dataclass
class ByteTrackArgs:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    mot20: bool = False


class ByteTrackTracker:
    def __init__(
        self,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        mot20: bool,
        frame_rate: int,
        line_thickness: int = 3,
    ):
        args = ByteTrackArgs(track_thresh, track_buffer, match_thresh, mot20)
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self.line_thickness = line_thickness

    def update(self, dets: np.ndarray):
        """
        dets: [[left, top, right, bottom, score], ...]
        """
        return "bbox", self.tracker.update(dets)

    def draw(self, image, tracks):
        for i, track in enumerate(tracks):
            # x1, y1, w, h = track.tlwh
            # ntbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            x1, y1, x2, y2 = track.tlbr
            obj_id = track.track_id
            color = self.get_color(abs(obj_id))
            cv2.rectangle(
                image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=color,
                thickness=self.line_thickness,
            )
        return image

    def get_color(self, label_id):
        color = (
            int((37 * label_id) % 255),
            int((17 * label_id) % 255),
            int((29 * label_id) % 255),
        )
        return color
