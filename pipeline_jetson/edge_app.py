"""Jetson edge app: GStreamer NV12 capture -> detect -> MQTT (detections only).

Tracking / map draw happen on the aggregation side so the edge can spend its
budget on detection. Optional debug display overlays detections on BGR frames.

Frames are never dropped (appsink backpressure). Optional prefetch overlaps
appsink→host NV12 copy with detector.infer().
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from pipeline_jetson.components.edge.gst_io import (
    GstBgrDisplay,
    GstNv12Capture,
    PrefetchNv12Capture,
    PullTimeout,
)
from pipeline_jetson.components.edge.publisher import Publisher


def _maybe_instantiate(obj):
    if isinstance(obj, DictConfig):
        return instantiate(obj)
    return obj


def nv12_to_bgr(nv12: np.ndarray, h: int, w: int) -> np.ndarray:
    yuv = nv12.reshape(h * 3 // 2, w)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def dets_to_payload(dets: np.ndarray) -> dict:
    """Convert Nx3 (x, y, score) detections to MQTT JSON-friendly dict."""
    points = []
    if dets is not None and len(dets) > 0:
        for row in np.asarray(dets, dtype=np.float64):
            points.append(
                {
                    "x": float(row[0]),
                    "y": float(row[1]),
                    "score": float(row[2]),
                }
            )
    return {"point": points, "bbox": []}


class PtsUnixEpochClock:
    """Map GStreamer PTS (relative) to Unix epoch while keeping PTS intervals.

    The first sample anchors wall-clock time; later samples are
    ``t0 + (pts - pts0)`` so frame spacing follows media time, not
    processing jitter.
    """

    def __init__(self) -> None:
        self._base_wall: float | None = None
        self._base_pts: float | None = None

    def reset(self) -> None:
        self._base_wall = None
        self._base_pts = None

    def to_unix(self, pts: float) -> float:
        if self._base_wall is None or self._base_pts is None:
            self._base_wall = time.time()
            self._base_pts = float(pts)
        return self._base_wall + (float(pts) - self._base_pts)


def draw_detections(
    bgr: np.ndarray,
    dets: np.ndarray,
    threshold: float = 0.5,
    point_size: int = 5,
    color=(0, 0, 255),
) -> np.ndarray:
    """Overlay point detections on a BGR frame (in-place)."""
    if dets is None or len(dets) == 0:
        return bgr
    for row in dets:
        x, y, score = int(row[0]), int(row[1]), row[2]
        if score < threshold:
            continue
        cv2.circle(bgr, (x, y), point_size, color, -1)
    return bgr


class EdgeApp:
    def __init__(
        self,
        source: str,
        size,
        detector,
        publisher=None,
        transport: str | None = None,
        max_buffers: int = 4,
        capture_fps=None,
        prefetch: bool = True,
        prefetch_queue_size: int = 2,
        display: bool = False,
        display_sink: str = "nveglglessink",
        display_threshold: float = 0.5,
        display_point_size: int = 5,
        display_color: tuple[int, int, int] = (0, 0, 255),
        report_every: int = 30,
        max_wall_seconds=None,
        max_frames=None,
    ):
        """
        Args:
            source: file path or rtsp:// URL
            size: [W, H] NV12 resolution (must match engine)
            detector: P2PNetTRTNV12Detector (or DictConfig)
            publisher: MQTT Publisher (detections payload)
            max_buffers: appsink depth; full => block, never drop under load
            capture_fps: intentional downsample via videorate (e.g. 15 from 30fps)
            prefetch: overlap next-frame host copy with detection
            prefetch_queue_size: prefetched frames held for the consumer
            display: if True, overlay dets and push to GStreamer sink (debug)
            display_sink: nveglglessink / nv3dsink / autovideosink / fakesink
            display_threshold: threshold for detection overlay
            display_point_size: circle radius for detection overlay
            display_color: color for detection overlay
            report_every: print FPS every N frames
            max_wall_seconds / max_frames: optional stop conditions
        """
        self.w, self.h = int(size[0]), int(size[1])
        raw = GstNv12Capture(
            source=source,
            size=(self.w, self.h),
            transport=transport,
            max_buffers=max_buffers,
            capture_fps=capture_fps,
        )
        self.prefetch = bool(prefetch)
        self.capture = (
            PrefetchNv12Capture(raw, queue_size=prefetch_queue_size)
            if self.prefetch
            else raw
        )
        self.detector = _maybe_instantiate(detector)
        self.publisher: Publisher | None = (
            _maybe_instantiate(publisher) if publisher is not None else None
        )
        self.display_enabled = bool(display)
        self.display_point_size = int(display_point_size)
        self.display_threshold = display_threshold
        self.display_color = display_color
        self.display = GstBgrDisplay(
            size=(self.w, self.h),
            sink=display_sink,
            enabled=self.display_enabled,
        )
        self.report_every = report_every
        self.max_wall_seconds = max_wall_seconds
        self.max_frames = max_frames
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self):
        self.capture.start()
        if self.display_enabled:
            self.display.start()
        if self.publisher is not None:
            self.publisher.start()

        if self.max_wall_seconds is not None and self.max_wall_seconds > 0:
            threading.Timer(self.max_wall_seconds, self.request_stop).start()
            print(f"Will stop after {self.max_wall_seconds}s")

        print(
            f"[edge] prefetch={'on' if self.prefetch else 'off'}  "
            f"display={'on' if self.display_enabled else 'off'}"
        )

        n = 0
        t_wait = t_det = t_pub = t_disp = 0.0
        t0_all = time.perf_counter()
        # Anchor once per run (capture start). PTS resets need a new run.
        epoch_clock = PtsUnixEpochClock()

        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    item = self.capture.pull()
                except PullTimeout:
                    # Live RTSP: keep waiting; file sources rarely hit this.
                    if getattr(self.capture, "is_rtsp", False):
                        print("[edge] waiting for RTSP frame...")
                        continue
                    print("[edge] pull timeout (file); stopping")
                    break
                t1 = time.perf_counter()
                if item is None:
                    print("[edge] EOS")
                    break

                nv12, seq, pts = item
                timestamp = epoch_clock.to_unix(pts)
                dets = self.detector.infer(nv12)
                t2 = time.perf_counter()

                if self.publisher is not None:
                    self.publisher.publish_detections(
                        seq, timestamp, dets_to_payload(dets)
                    )
                t3 = time.perf_counter()

                if self.display_enabled:
                    bgr = nv12_to_bgr(nv12, self.h, self.w)
                    draw_detections(
                        bgr,
                        dets,
                        threshold=self.display_threshold,
                        point_size=self.display_point_size,
                        color=self.display_color,
                    )
                    if not self.display.push(bgr):
                        print("[edge] display push failed; stopping")
                        break
                t4 = time.perf_counter()

                n += 1
                t_wait += t1 - t0
                t_det += t2 - t1
                t_pub += t3 - t2
                t_disp += t4 - t3

                if self.report_every > 0 and n % self.report_every == 0:
                    e2e = (t4 - t0) * 1e3
                    avg = (time.perf_counter() - t0_all) / n
                    msg = (
                        f"[edge] seq={seq}  "
                        f"wait={(t1 - t0) * 1e3:.1f}  "
                        f"det={(t2 - t1) * 1e3:.1f}  "
                        f"pub={(t3 - t2) * 1e3:.1f}  "
                    )
                    if self.display_enabled:
                        msg += f"disp={(t4 - t3) * 1e3:.1f}  "
                    msg += (
                        f"e2e={e2e:.1f}ms  "
                        f"avg={avg * 1e3:.1f}ms ({1 / avg:.1f}fps)  "
                        f"dets={0 if dets is None else len(dets)}"
                    )
                    print(msg)

                if self.max_frames is not None and n >= self.max_frames:
                    print(f"[edge] reached max_frames={self.max_frames}")
                    break
        finally:
            if n:
                summary = (
                    f"[edge] done frames={n}  "
                    f"avg wait={t_wait / n * 1e3:.1f}  "
                    f"det={t_det / n * 1e3:.1f}  "
                    f"pub={t_pub / n * 1e3:.1f}"
                )
                if self.display_enabled:
                    summary += f"  disp={t_disp / n * 1e3:.1f}"
                summary += (
                    f" ms  prefetch={'on' if self.prefetch else 'off'}  "
                    f"display={'on' if self.display_enabled else 'off'}"
                )
                print(summary)
            if self.display_enabled:
                self.display.stop()
            self.capture.stop()
            if self.publisher is not None:
                self.publisher.stop()
