"""GStreamer I/O for Jetson edge: NV12 capture + BGR display.

Capture (no frame drop):
  demux/parse -> nvv4l2decoder -> nvvidconv -> NV12 appsink
  appsink: drop=false, max-buffers=N  -> when full, upstream blocks
  (backpressure) instead of discarding frames.

Display:
  appsrc (BGR) -> videoconvert -> nvvidconv -> nveglglessink
  (or fakesink when display is disabled / headless).
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # noqa: E402

from pipeline_jetson.components.edge.nv12_homography import (
    Nv12HomographyWarper,
    load_homography,
)


def _ensure_gst():
    if not Gst.is_initialized():
        Gst.init(None)


class PullTimeout(Exception):
    """appsink had no sample within the timeout (not EOS)."""


class GstNv12Capture:
    """Blocking NV12 frame source. Pulls frames in order.

    By default appsink uses drop=false (backpressure, no skip under load).
    Optional capture_fps inserts videorate to intentionally downsample
    (e.g. 30fps stream -> 15fps for detection).
    """

    def __init__(
        self,
        source: str,
        size: Tuple[int, int],
        transport: Optional[str] = None,
        max_buffers: int = 4,
        capture_fps: Optional[float] = None,
        calibration_enabled: bool = False,
        calibration_homography_path: Optional[str] = None,
    ):
        """
        Args:
            source: file path, rtsp:// URL, or /dev/video* (USB)
            size: (W, H) NV12 output size (must match TRT engine)
            transport: RTSP transport ("tcp"/"udp"); required for rtsp
            max_buffers: appsink queue depth. When full, decoder blocks
                until pull() — frames are never discarded under load.
            capture_fps: if set, keep frames by PTS so output ≈ this rate
                (e.g. 15 from a 30fps camera). Done in pull(), not videorate —
                videorate stalls this RTSP+NV12 path on Jetson.
            calibration_enabled: if True, apply homography warp after capture
            calibration_homography_path: 3x3 matrix text file for calibration
        """
        _ensure_gst()
        self.source = source
        self.is_rtsp = source.lower().startswith(("rtsp://", "rtsps://"))
        self.is_usb = source.startswith("/dev/video")
        self.w, self.h = int(size[0]), int(size[1])
        self.transport = transport
        self.max_buffers = max(1, int(max_buffers))
        self.capture_fps = (
            float(capture_fps) if capture_fps is not None and capture_fps > 0 else None
        )
        self._min_frame_dt = (
            1.0 / self.capture_fps if self.capture_fps is not None else None
        )
        self.nv12_nbytes = self.w * self.h * 3 // 2

        self._pipe = None
        self._sink: Optional[GstApp.AppSink] = None
        self._seq = 0
        self._eos = False
        self._last_kept_pts: Optional[float] = None
        self._fps_note = "capture_fps=native"
        self.calibration_enabled = bool(calibration_enabled)
        self._warper: Optional[Nv12HomographyWarper] = None
        self._warp_out = np.empty(self.nv12_nbytes, dtype=np.uint8)
        if self.calibration_enabled:
            if not calibration_homography_path:
                raise ValueError(
                    "calibration_homography_path is required when calibration_enabled=true"
                )
            homography = load_homography(calibration_homography_path)
            self._warper = Nv12HomographyWarper(
                homography=homography,
                width=self.w,
                height=self.h,
            )
            self._calib_note = f"calibration=on path={calibration_homography_path}"
        else:
            self._calib_note = "calibration=off"

    def _nv12_tail(self, caps: str) -> str:
        return (
            f"{caps} ! appsink name=sink emit-signals=false "
            f"max-buffers={self.max_buffers} drop=false sync=false"
        )

    def _build_desc(self) -> str:
        # parsebin auto-picks h264parse / h265parse (and matching depay for RTSP).
        if self.is_rtsp:
            if self.transport is None:
                raise ValueError("transport is required for RTSP source")
            # latency>0 helps first frames arrive on some cameras (i-PRO etc.)
            src = (
                f"rtspsrc location={self.source} protocols={self.transport} "
                f"latency=200 ! parsebin"
            )
        elif self.is_usb:
            if not Path(self.source).exists():
                raise FileNotFoundError(f"USB video device not found: {self.source}")
            src = f"v4l2src device={self.source}"
        else:
            if not Path(self.source).exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")
            src = f"filesrc location={self.source} ! qtdemux ! parsebin"

        caps = f"video/x-raw,format=NV12,width={self.w},height={self.h}"
        if self.capture_fps is not None:
            fps_note = f"capture_fps={self.capture_fps:g} (pts-skip)"
        else:
            fps_note = "capture_fps=native"
        # drop=false + finite max-buffers => backpressure under load
        # sync=false => pull paced by the consumer, not the pipeline clock
        self._fps_note = fps_note
        if self.is_usb:
            # UVC / V4L2 cameras typically output in system memory; keep it simple
            # and convert to packed NV12 in CPU space for appsink.
            return f"{src} ! videoconvert ! {self._nv12_tail(caps)}"
        if self.calibration_enabled:
            # Decode/convert on NVMM, then download once before appsink. Perspective
            # warp itself runs on CUDA in pull() to avoid a BGR round-trip.
            nvmm_caps = (
                f"video/x-raw(memory:NVMM),format=NV12,width={self.w},height={self.h}"
            )
            return (
                f"{src} ! nvv4l2decoder ! nvvidconv ! {nvmm_caps} ! "
                f"nvvidconv ! {self._nv12_tail(caps)}"
            )
        return f"{src} ! nvv4l2decoder ! nvvidconv ! {self._nv12_tail(caps)}"

    def start(self) -> None:
        self.stop()
        desc = self._build_desc()
        pipe = Gst.parse_launch(desc)
        sink = pipe.get_by_name("sink")
        assert sink is not None
        pipe.set_state(Gst.State.PLAYING)
        self._pipe = pipe
        self._sink = sink
        self._seq = 0
        self._eos = False
        self._last_kept_pts = None
        print(
            f"[GST-capture] started  {self.w}x{self.h} NV12  "
            f"max-buffers={self.max_buffers} drop=false  {self._fps_note}  "
            f"{self._calib_note}"
        )

    def stop(self) -> None:
        pipe = self._pipe
        self._pipe = None
        self._sink = None
        if pipe is not None:
            pipe.set_state(Gst.State.NULL)

    def _poll_bus(self) -> Optional[str]:
        """Return 'eos' / 'error' if present on the bus, else None."""
        if self._pipe is None:
            return None
        bus = self._pipe.get_bus()
        msg = bus.pop_filtered(Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg is None:
            return None
        if msg.type == Gst.MessageType.EOS:
            self._eos = True
            return "eos"
        err, debug = msg.parse_error()
        print(f"[GST-capture] ERROR: {err.message} ({debug})")
        self._eos = True
        return "error"

    def _try_pull_sample(self, timeout_s: float):
        """Pull one appsink sample. None=EOS/error; raises PullTimeout."""
        if self._sink is None or self._eos:
            return None
        sample = self._sink.try_pull_sample(int(timeout_s * Gst.SECOND))
        if sample is None:
            status = self._poll_bus()
            if status in ("eos", "error"):
                return None
            raise PullTimeout(f"no frame within {timeout_s:.1f}s")
        return sample

    @staticmethod
    def _pts_seconds(buf) -> Optional[float]:
        pts = buf.pts
        if pts == Gst.CLOCK_TIME_NONE:
            return None
        return float(pts) / Gst.SECOND

    def _copy_mapped_nv12(self, buf) -> np.ndarray:
        """Map buffer once, copy NV12 bytes, optional calibration warp."""
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise PullTimeout("buffer map failed")
        try:
            if mapinfo.size < self.nv12_nbytes:
                raise PullTimeout(
                    f"short buffer {mapinfo.size} < {self.nv12_nbytes}"
                )
            frame = np.frombuffer(
                mapinfo.data, dtype=np.uint8, count=self.nv12_nbytes
            ).copy()
        finally:
            buf.unmap(mapinfo)

        if self._warper is not None:
            self._warper.warp(frame, out=self._warp_out)
            return self._warp_out.copy()
        return frame

    def pull(
        self, timeout_s: float = 5.0
    ) -> Optional[Tuple[np.ndarray, int, Optional[float]]]:
        """Pull next kept NV12 frame (after optional capture_fps skip).

        Returns (nv12_copy, seq, pts_seconds_or_None).
        PTS is None when the buffer has Gst.CLOCK_TIME_NONE (kept, not fps-skipped).
        Returns None on EOS / fatal error.
        Raises PullTimeout if no kept frame arrived within timeout_s.
        """
        deadline = time.monotonic() + float(timeout_s)
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise PullTimeout(f"no kept frame within {timeout_s:.1f}s")
            sample = self._try_pull_sample(timeout_s=remain)
            if sample is None:
                return None
            buf = sample.get_buffer()
            pts_s = self._pts_seconds(buf)
            # PTS-first skip: avoid map/copy on frames we will drop.
            # Missing PTS is always kept (cannot safely throttle).
            if (
                pts_s is not None
                and self._min_frame_dt is not None
                and self._last_kept_pts is not None
                and (pts_s - self._last_kept_pts) < self._min_frame_dt * 0.85
            ):
                continue
            nv12 = self._copy_mapped_nv12(buf)
            self._last_kept_pts = pts_s
            self._seq += 1
            return nv12, self._seq, pts_s


class PrefetchNv12Capture:
    """Background pull so host NV12 copy overlaps with detection.

    GStreamer already decodes into appsink while the consumer is busy; this
    additionally hides the appsink→host memcpy by running it on a worker
    thread during detector.infer().
    """

    def __init__(self, capture: GstNv12Capture, queue_size: int = 2):
        self._cap = capture
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def is_rtsp(self) -> bool:
        return self._cap.is_rtsp

    @property
    def is_usb(self) -> bool:
        return self._cap.is_usb

    @property
    def w(self) -> int:
        return self._cap.w

    @property
    def h(self) -> int:
        return self._cap.h

    def start(self) -> None:
        self.stop()
        self._stop.clear()
        self._cap.start()
        self._thread = threading.Thread(
            target=self._worker, name="nv12-prefetch", daemon=True
        )
        self._thread.start()
        print(f"[GST-prefetch] queue_size={self._q.maxsize}")

    def stop(self) -> None:
        self._stop.set()
        # Unblock worker stuck in queue.put / pull
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._cap.stop()
        # Drain again after stop
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _worker(self) -> None:
        idle_logs = 0
        while not self._stop.is_set():
            try:
                item = self._cap.pull(timeout_s=2.0)
            except PullTimeout:
                idle_logs += 1
                if idle_logs == 1 or idle_logs % 5 == 0:
                    print(
                        f"[GST-prefetch] waiting for first/next frame "
                        f"({idle_logs * 2}s)..."
                    )
                continue
            if self._stop.is_set():
                break
            idle_logs = 0
            # Always enqueue (including None = EOS) so consumer can exit.
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if item is None:
                print("[GST-prefetch] EOS/error; worker exit")
                break

    def pull(
        self, timeout_s: float = 5.0
    ) -> Optional[Tuple[np.ndarray, int, Optional[float]]]:
        """Get next prefetched frame (or None on EOS). Raises PullTimeout."""
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            raise PullTimeout(f"prefetch queue empty for {timeout_s:.1f}s")


class GstBgrDisplay:
    """Push BGR frames into a GStreamer display pipeline."""

    def __init__(
        self,
        size: Tuple[int, int],
        sink: str = "nveglglessink",
        enabled: bool = True,
    ):
        """
        Args:
            size: (W, H) of BGR frames pushed to appsrc
            sink: e.g. nveglglessink / nv3dsink / autovideosink / fakesink
            enabled: if False, uses fakesink (no window)
        """
        _ensure_gst()
        self.w, self.h = int(size[0]), int(size[1])
        self.sink_name = sink if enabled else "fakesink"
        self.enabled = enabled
        self._pipe = None
        self._appsrc: Optional[GstApp.AppSrc] = None
        self._frame_bytes = self.w * self.h * 3

    def _build_desc(self) -> str:
        caps = (
            f"video/x-raw,format=BGR,width={self.w},height={self.h},"
            f"framerate=0/1"
        )
        # Jetson: nveglglessink cannot consume NVMM surface-array buffers
        # directly ("eglglessink cannot handle NVRM surface array ...").
        # Insert nvegltransform (NVMM -> EGLImage) before the sink.
        if self.sink_name == "nveglglessink":
            return (
                f"appsrc name=src is-live=true format=time do-timestamp=true "
                f"caps={caps} ! "
                f"videoconvert ! video/x-raw,format=RGBA ! "
                f"nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! "
                f"nvegltransform ! "
                f"nveglglessink name=vsink sync=false"
            )
        if self.sink_name == "nv3dsink":
            return (
                f"appsrc name=src is-live=true format=time do-timestamp=true "
                f"caps={caps} ! "
                f"videoconvert ! video/x-raw,format=RGBA ! "
                f"nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! "
                f"nv3dsink name=vsink sync=false"
            )
        # CPU sinks (ximagesink / xvimagesink / autovideosink / fakesink)
        return (
            f"appsrc name=src is-live=true format=time do-timestamp=true "
            f"caps={caps} ! videoconvert ! "
            f"{self.sink_name} name=vsink sync=false"
        )

    def start(self) -> None:
        self.stop()
        desc = self._build_desc()
        pipe = Gst.parse_launch(desc)
        src = pipe.get_by_name("src")
        assert src is not None
        src.set_property("block", True)  # backpressure if display is slow
        pipe.set_state(Gst.State.PLAYING)
        self._pipe = pipe
        self._appsrc = src
        print(f"[GST-display] started  {self.w}x{self.h} BGR -> {self.sink_name}")

    def stop(self) -> None:
        pipe = self._pipe
        self._pipe = None
        self._appsrc = None
        if pipe is not None:
            pipe.set_state(Gst.State.NULL)

    def push(self, bgr: np.ndarray) -> bool:
        """Push one BGR frame. Returns False if appsrc is gone / flush failed."""
        if self._appsrc is None:
            return False
        assert bgr.shape == (self.h, self.w, 3), (
            f"expected {(self.h, self.w, 3)}, got {bgr.shape}"
        )
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        buf = Gst.Buffer.new_allocate(None, self._frame_bytes, None)
        buf.fill(0, bgr.tobytes())
        ret = self._appsrc.emit("push-buffer", buf)
        return ret == Gst.FlowReturn.OK
