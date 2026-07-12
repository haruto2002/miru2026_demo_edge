from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

LABEL_MAP_ID_TO_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class YOLODetector:
    def __init__(
        self,
        model_size: str,
        weight_dir: str,
        device: str,
        img_long_side: int | None,
        conf_threshold: float,
        classes: list[int] | None = None,
    ):
        assert model_size in ["n", "s", "m", "l", "x"], "Invalid model size"
        self.weight_path = Path(weight_dir) / f"yolo26{model_size}.pt"
        self.device = device
        self.img_long_side = img_long_side
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.model = self.build_detector()

    def build_detector(self):
        model = YOLO(self.weight_path)
        model.to(self.device)
        model.eval()
        return model

    def infer(self, image: np.ndarray):
        if self.img_long_side is None:
            height, width = image.shape[:2]
            self.img_long_side = max(height, width)
        result = self.model(
            image,
            imgsz=self.img_long_side,
            conf=self.conf_threshold,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        labels = result.boxes.cls.detach().cpu().numpy()
        result = np.concatenate(
            [
                boxes,
                np.expand_dims(labels, -1),
                np.expand_dims(scores, -1),
            ],
            axis=1,
        )
        return result

    def display_result(self, image: np.ndarray, result: np.ndarray):
        for left, top, right, bottom, label, score in result:
            left, top, right, bottom = map(int, [left, top, right, bottom])
            color = self.get_color(label)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            label_name = self.model.names[int(label)]
            cv2.putText(
                image,
                label_name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3,
            )
        return image

    def get_color(self, label_id):
        color = (
            int((37 * label_id) % 255),
            int((17 * label_id) % 255),
            int((29 * label_id) % 255),
        )
        return color
