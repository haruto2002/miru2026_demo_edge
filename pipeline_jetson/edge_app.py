"""Jetson edge app: GStreamer NV12 capture -> detect -> MQTT (detections only).

Tracking / map draw happen on the aggregation side so the edge can spend its
budget on detection. Optional debug display overlays detections on BGR frames
on a background thread so convert/draw/push do not stall detect or MQTT.

Frames are never dropped (appsink backpressure). Optional prefetch overlaps
appsink→host NV12 copy with detector.infer().
"""

from __future__ import annotations

import queue
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
from pipeline_jetson.components.edge.log_util import get_logger
from pipeline_jetson.components.edge.publisher import Publisher

log = get_logger(__name__)


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

    The first sample with a valid PTS anchors wall-clock time; later samples are
    ``t0 + (pts - pts0)`` so frame spacing follows media time, not
    processing jitter. ``pts is None`` returns ``time.time()`` without
    updating the anchor.
    """

    def __init__(self) -> None:
        self._base_wall: float | None = None
        self._base_pts: float | None = None

    def reset(self) -> None:
        self._base_wall = None
        self._base_pts = None

    def to_unix(self, pts: float | None) -> float:
        # Missing media PTS: wall clock only; do not disturb the PTS anchor.
        if pts is None:
            return time.time()
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


class AsyncDetectionDisplay:
    """Latest-only display worker: NV12+dets -> BGR overlay -> GstBgrDisplay.

    Depth-1 queue; submit never blocks the detect/MQTT path. When display is
    slower than inference, older pending frames are dropped.
    """

    def __init__(
        self,
        size: tuple[int, int],
        sink: str = "nveglglessink",
        threshold: float = 0.5,
        point_size: int = 5,
        color: tuple[int, int, int] = (0, 0, 255),
    ):
        self.w, self.h = int(size[0]), int(size[1])
        self.threshold = float(threshold)
        self.point_size = int(point_size)
        self.color = color
        self._display = GstBgrDisplay(size=(self.w, self.h), sink=sink, enabled=True)
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.push_failed = False

    def start(self) -> None:
        self.stop()
        self._stop.clear()
        self.push_failed = False
        self._display.start()
        self._thread = threading.Thread(
            target=self._worker, name="det-display", daemon=True
        )
        self._thread.start()
        log.info("[edge] async display started (queue=1, latest-only)")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._display.stop()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def submit(self, nv12: np.ndarray, dets: np.ndarray) -> None:
        """Enqueue latest frame; drop any pending job. Never blocks."""
        item = (nv12, dets)
        try:
            self._q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            nv12, dets = item
            bgr = nv12_to_bgr(nv12, self.h, self.w)
            draw_detections(
                bgr,
                dets,
                threshold=self.threshold,
                point_size=self.point_size,
                color=self.color,
            )
            if not self._display.push(bgr):
                self.push_failed = True
                log.info("[edge] display push failed; display worker exit")
                break


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
        drop_frames_when_lagging: bool = False,
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
        calibration_enabled: bool = False,
        calibration_homography_path: str | None = None,
    ):
        """
        Args:
            source: file path or rtsp:// URL
            size: [W, H] NV12 resolution (must match engine)
            detector: P2PNetTRTNV12Detector (or DictConfig)
            publisher: MQTT Publisher (detections payload)
            max_buffers: appsink depth; full => block, never drop under load
            capture_fps: intentional downsample via videorate (e.g. 15 from 30fps)
            drop_frames_when_lagging: if True, appsink drops older frames when
                full to keep the pipeline near real time
            prefetch: overlap next-frame host copy with detection
            prefetch_queue_size: prefetched frames held for the consumer
            display: if True, async overlay of dets on a background thread
            display_sink: nveglglessink / nv3dsink / autovideosink / fakesink
            display_threshold: threshold for detection overlay
            display_point_size: circle radius for detection overlay
            display_color: color for detection overlay
            report_every: print FPS every N frames
            max_wall_seconds / max_frames: optional stop conditions
            calibration_enabled: apply homography warp to captured NV12
            calibration_homography_path: 3x3 homography text file
        """
        self.w, self.h = int(size[0]), int(size[1])
        raw = GstNv12Capture(
            source=source,
            size=(self.w, self.h),
            transport=transport,
            max_buffers=max_buffers,
            capture_fps=capture_fps,
            drop_frames_when_lagging=drop_frames_when_lagging,
            calibration_enabled=calibration_enabled,
            calibration_homography_path=calibration_homography_path,
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
        self.async_display: AsyncDetectionDisplay | None = (
            AsyncDetectionDisplay(
                size=(self.w, self.h),
                sink=display_sink,
                threshold=display_threshold,
                point_size=display_point_size,
                color=display_color,
            )
            if self.display_enabled
            else None
        )
        self.report_every = report_every
        self.max_wall_seconds = max_wall_seconds
        self.max_frames = max_frames
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self):
        self.capture.start()
        if self.async_display is not None:
            self.async_display.start()
        if self.publisher is not None:
            self.publisher.start()

        if self.max_wall_seconds is not None and self.max_wall_seconds > 0:
            threading.Timer(self.max_wall_seconds, self.request_stop).start()
            log.info("Will stop after %ss", self.max_wall_seconds)

        log.info(
            "[edge] prefetch=%s  display=%s",
            "on" if self.prefetch else "off",
            "on" if self.display_enabled else "off",
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
                    # Live sources (RTSP/USB): keep waiting; file sources rarely hit this.
                    if getattr(self.capture, "is_rtsp", False) or getattr(
                        self.capture, "is_usb", False
                    ):
                        log.info("[edge] waiting for live frame...")
                        continue
                    log.info("[edge] pull timeout (file); stopping")
                    break
                t1 = time.perf_counter()
                if item is None:
                    log.info("[edge] EOS")
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

                if self.async_display is not None:
                    self.async_display.submit(nv12, dets)
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
                        f"[edge] seq={seq}  ts={timestamp:.3f}  "
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
                    log.info(msg)

                if self.max_frames is not None and n >= self.max_frames:
                    log.info("[edge] reached max_frames=%s", self.max_frames)
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
                log.info(summary)
            if self.async_display is not None:
                self.async_display.stop()
            self.capture.stop()
            if self.publisher is not None:
                self.publisher.stop()
