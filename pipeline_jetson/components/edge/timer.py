from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class WorkerTimeStamps(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: Optional[float] = None
    frame_set: Optional[float] = None
    detected: Optional[float] = None
    tracked: Optional[float] = None
    drawn: Optional[float] = None
    submitted: Optional[float] = None
    end: Optional[float] = None


class FrameReaderTimeStamps(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: Optional[float] = None
    arrived: Optional[float] = None
    loaded: Optional[float] = None
    end: Optional[float] = None


class DisplayerTimeStamps(BaseModel):
    model_config = ConfigDict(extra="ignore")

    arrived: Optional[float] = None
    start: Optional[float] = None
    frame_set: Optional[float] = None
    displayed: Optional[float] = None
    end: Optional[float] = None


class TimeStamps(BaseModel):
    model_config = ConfigDict(extra="ignore")

    worker: WorkerTimeStamps = Field(default_factory=WorkerTimeStamps)
    frame_reader: FrameReaderTimeStamps = Field(default_factory=FrameReaderTimeStamps)
    displayer: DisplayerTimeStamps = Field(default_factory=DisplayerTimeStamps)

    frame_read_fps: Optional[float] = None
    frame_set_fps: Optional[float] = None
    submit_fps: Optional[float] = None
    display_fps: Optional[float] = None


_FPSKey = Literal["frame_read_fps", "frame_set_fps", "submit_fps", "display_fps"]
_LogAdapter = TypeAdapter(Dict[int, TimeStamps])


class TimeCounter:
    def __init__(self, save_dir: str | None = None) -> None:
        self.log: dict[int, TimeStamps] = {}
        self.save_dir = save_dir

    def add(self, seq: int) -> TimeStamps:
        ts = TimeStamps()
        self.log[seq] = ts
        return ts

    def get(self, seq: int) -> TimeStamps:
        try:
            return self.log[seq]
        except KeyError as e:
            raise KeyError(f"Sequence {seq} is not recorded") from e

    @staticmethod
    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return a - b

    @staticmethod
    def _fmt_ms(x: float | None) -> str:
        return "—" if x is None else f"{round(x * 1000)}"

    @staticmethod
    def _label(name: str, width: int = 18) -> str:
        return f"{name:<{width}}"

    def latency(self, seq: int) -> float | None:
        ts = self.get(seq)
        return self._delta(ts.displayer.displayed, ts.frame_reader.arrived)

    def frame_set_latency(self, seq: int) -> tuple[float | None, float | None]:
        ts = self.get(seq)
        process_time = self._delta(ts.frame_reader.end, ts.frame_reader.arrived)
        waiting_time = self._delta(ts.worker.frame_set, ts.frame_reader.end)
        return process_time, waiting_time

    def det_latency(self, seq: int) -> float | None:
        ts = self.get(seq)
        return self._delta(ts.worker.detected, ts.worker.frame_set)

    def track_latency(self, seq: int) -> float | None:
        ts = self.get(seq)
        return self._delta(ts.worker.tracked, ts.worker.detected)

    def draw_latency(self, seq: int) -> float | None:
        ts = self.get(seq)
        return self._delta(ts.worker.drawn, ts.worker.tracked)

    def display_latency(self, seq: int) -> tuple[float | None, float | None]:
        ts = self.get(seq)
        process_time = self._delta(ts.displayer.displayed, ts.displayer.frame_set)
        waiting_time = self._delta(ts.displayer.frame_set, ts.displayer.arrived)
        return process_time, waiting_time

    def make_single_report(self, seq: int) -> str:
        frame_p, frame_w = self.frame_set_latency(seq)
        display_p, display_w = self.display_latency(seq)

        lines = [
            "",
            f"=================== SEQ: {seq} =====================",
            f"{self._label('Latency')}: {self._fmt_ms(self.latency(seq)):>4} ms",
            f"{self._label('Frame set latency')}: process {self._fmt_ms(frame_p):>4} ms, waiting {self._fmt_ms(frame_w):>4} ms",
            f"{self._label('Det latency')}: {self._fmt_ms(self.det_latency(seq)):>4} ms",
            f"{self._label('Track latency')}: {self._fmt_ms(self.track_latency(seq)):>4} ms",
            f"{self._label('Draw latency')}: {self._fmt_ms(self.draw_latency(seq)):>4} ms",
            f"{self._label('Display latency')}: waiting {self._fmt_ms(display_w):>4} ms, process {self._fmt_ms(display_p):>4} ms",
        ]
        return "\n".join(lines)

    def report_single(self, seq: int) -> None:
        print(self.make_single_report(seq))

    def _avg_window_items(
        self, current_seq: int, num_frames: int
    ) -> list[tuple[int, TimeStamps]]:
        """current_seq が直近処理完了側のとき、末尾 num_frames はみのログを取る。"""
        if num_frames <= 0:
            raise ValueError("num_frames must be greater than 0")
        items = sorted(self.log.items())
        start_idx = max(0, current_seq - num_frames)
        return items[start_idx : current_seq]

    def calc_avg_fps(
        self, current_seq: int, num_frames: int, fps_key: _FPSKey
    ) -> float:
        if num_frames <= 0:
            raise ValueError("num_frames must be greater than 0")

        fps_values = [
            value
            for _, data in self._avg_window_items(current_seq, num_frames)
            if (value := getattr(data, fps_key)) is not None and value > 0
        ]
        return fmean(fps_values) if fps_values else 0.0

    def calc_avg_latency(
        self,
        current_seq: int,
        num_frames: int,
        calculater,
        with_queue: bool = False,
    ) -> tuple[float, float | None]:
        if num_frames <= 0:
            raise ValueError("num_frames must be greater than 0")

        seq_list = [seq for seq, _ in self._avg_window_items(current_seq, num_frames)]
        if not with_queue:
            process_times = [
                v for seq in seq_list if (v := calculater(seq)) is not None
            ]
            return fmean(process_times) if process_times else 0.0, None
        else:
            process_times = [
                v for seq in seq_list if (v := calculater(seq)[0]) is not None
            ]
            waiting_times = [
                v for seq in seq_list if (v := calculater(seq)[1]) is not None
            ]
            return fmean(process_times) if process_times else 0.0, fmean(
                waiting_times
            ) if waiting_times else 0.0

    def make_avg_fps_report(self, current_seq: int, num_frames: int) -> list[str]:
        lines = [
            "< FPS >",
            f"{self._label('Frame read')}: {self.calc_avg_fps(current_seq, num_frames, 'frame_read_fps'):.2f}",
            f"{self._label('Frame set')}: {self.calc_avg_fps(current_seq, num_frames, 'frame_set_fps'):.2f}",
            f"{self._label('Submit')}: {self.calc_avg_fps(current_seq, num_frames, 'submit_fps'):.2f}",
            f"{self._label('Display')}: {self.calc_avg_fps(current_seq, num_frames, 'display_fps'):.2f}",
        ]
        return lines

    def make_avg_latency_report(self, current_seq: int, num_frames: int) -> list[str]:
        total, _ = self.calc_avg_latency(current_seq, num_frames, self.latency)
        avg_frame_set, avg_frame_set_waiting = self.calc_avg_latency(
            current_seq, num_frames, self.frame_set_latency, with_queue=True
        )
        avg_det, _ = self.calc_avg_latency(current_seq, num_frames, self.det_latency)
        avg_track, _ = self.calc_avg_latency(current_seq, num_frames, self.track_latency)
        avg_draw, _ = self.calc_avg_latency(current_seq, num_frames, self.draw_latency)
        avg_display, avg_display_waiting = self.calc_avg_latency(
            current_seq, num_frames, self.display_latency, with_queue=True
        )
        lines = [
            "< Latency >",
            f"{self._label('Latency')}: {self._fmt_ms(total):>4} ms",
            f"{self._label('Frame set latency')}: process {self._fmt_ms(avg_frame_set):>4} ms, waiting {self._fmt_ms(avg_frame_set_waiting):>4} ms",
            f"{self._label('Det latency')}: {self._fmt_ms(avg_det):>4} ms",
            f"{self._label('Track latency')}: {self._fmt_ms(avg_track):>4} ms",
            f"{self._label('Draw latency')}: {self._fmt_ms(avg_draw):>4} ms",
            f"{self._label('Display latency')}: waiting {self._fmt_ms(avg_display_waiting):>4} ms, process {self._fmt_ms(avg_display):>4} ms",
        ]
        return lines

    def make_avg_report(self, current_seq: int, num_frames: int) -> str:
        fps_lines = self.make_avg_fps_report(current_seq, num_frames)
        latency_lines = self.make_avg_latency_report(current_seq, num_frames)

        first_line = (
            f"~~~~~~~~~~~~~~~~~ Average (last {num_frames} frames) ~~~~~~~~~~~~~~~~~~"
        )
        last_line = "~" * len(first_line)

        lines = [
            "",
            first_line,
            *fps_lines,
            "",
            *latency_lines,
            last_line,
        ]
        return "\n".join(lines)

    def report_avg(self, current_seq: int, num_frames: int) -> None:
        print(self.make_avg_report(current_seq, num_frames))

    def save(self, save_dir: str | None = None) -> None:
        if save_dir is not None:
            self.save_dir = save_dir
        elif self.save_dir is None:
            print("No save directory provided")
            return
        log_path = Path(self.save_dir) / "time_counter.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            _LogAdapter.dump_json(self.log, indent=4).decode("utf-8") + "\n",
            encoding="utf-8",
        )

        summary_path = Path(self.save_dir) / "summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            self.make_avg_report(max(self.log.keys(), default=0), 100),
            encoding="utf-8",
        )


class TimeAnalyzer(TimeCounter):
    def __init__(self, json_file: str | Path | None = None) -> None:
        super().__init__()
        self.load_timing_log(json_file)
        self.remove_data_with_null()

    def remove_data_with_null(self) -> None:
        """いずれかのフィールドが None の seq は log から削除する。"""
        self.log = {
            seq: ts
            for seq, ts in self.log.items()
            if not self._value_tree_contains_none(ts.model_dump(mode="python"))
        }

    def load_timing_log(self, load_path: str | Path | None = None) -> None:
        if load_path is None:
            return

        path = Path(load_path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            raise ValueError("Invalid timing log: root must be a JSON object")

        if "log" in raw and isinstance(raw["log"], dict):
            raw = raw["log"]

        filtered: dict[int, Any] = {}
        for key, val in raw.items():
            if not isinstance(val, dict):
                continue
            try:
                filtered[int(key)] = val
            except (TypeError, ValueError):
                continue

        self.log = _LogAdapter.validate_python(filtered)

    def _value_tree_contains_none(self, obj: Any) -> bool:
        """dict / list / ネストした構造のいずれかに None があれば True。"""
        if obj is None:
            return True
        if isinstance(obj, dict):
            return any(self._value_tree_contains_none(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(self._value_tree_contains_none(x) for x in obj)
        return False

    def summary(self, num_frames: int) -> None:
        if not self.log:
            return
        self.report_avg(max(self.log.keys()), num_frames)


if __name__ == "__main__":
    ta = TimeAnalyzer("time_counter.json")
    ta.summary(100)
