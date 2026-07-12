from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from processor.detector.p2pnet import P2PNetDetector
from processor.detector.yolo26 import YOLODetector
from processor.tracker.bytetrack import ByteTrackTracker
from processor.tracker.point_bytetrack import PointByteTrackTracker


class Detector:
    def __init__(
        self,
        detector: P2PNetDetector | YOLODetector | Path,
        devide_size: tuple[int, int] | None = None,
    ):
        if isinstance(detector, Path):
            cfg = OmegaConf.load(detector)
            self.detector = instantiate(cfg)
        else:
            self.detector = detector
        self.devide_size = devide_size

    def infer(self, image: np.ndarray) -> np.ndarray:
        if self.devide_size is None:
            return self.detector.infer(image)
        else:
            return self.infer_split(image, *self.devide_size)

    def infer_split(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        images, positions = devide_image(image, height, width)
        results = self.detector.infer_batch(images)
        merged_results = merge_results(results, positions)
        return merged_results

    def draw(self, image: np.ndarray, dets: np.ndarray) -> np.ndarray:
        return self.detector.display_result(image, dets)


class Tracker:
    def __init__(self, tracker: PointByteTrackTracker | ByteTrackTracker | Path):
        if isinstance(tracker, Path):
            cfg = OmegaConf.load(tracker)
            self.tracker = instantiate(cfg)
        else:
            self.tracker = tracker

    def update(self, dets: np.ndarray) -> np.ndarray:
        return self.tracker.update(dets)

    def convert_to_tracker_inputs(self, dets):
        if dets.shape[1] == 3:  # point(x, y, score)
            return dets
        elif dets.shape[1] == 6:  # bbox(left, top, right, bottom, label, score)
            dets = [dets[:, :4], dets[:, 5:6]]
            dets = np.concatenate(dets, axis=1)
            return dets
        else:
            raise ValueError(f"Invalid detection shape: {dets.shape}")

    def draw(self, image, tracks):
        return self.tracker.draw(image, tracks)


def devide_image(
    image: np.ndarray, height: int, width: int
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    h_num = (image.shape[0] + height - 1) // height
    w_num = (image.shape[1] + width - 1) // width
    images = []
    positions = []
    for i in range(h_num):
        for j in range(w_num):
            images.append(
                image[i * height : (i + 1) * height, j * width : (j + 1) * width]
            )
            positions.append((i * height, j * width))
    return images, positions


def merge_results(results, positions) -> np.ndarray:
    merged_results = []
    if results[0].shape[1] == 6:
        for result, position in zip(results, positions):
            result[:, 0] += position[1]
            result[:, 2] += position[1]
            result[:, 1] += position[0]
            result[:, 3] += position[0]
            merged_results.append(result)
    else:
        raise ValueError(f"Invalid result shape: {results[0].shape}")
    return np.concatenate(merged_results, axis=0)
