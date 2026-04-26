"""Pupil Labs Visualization Syntalos Module."""

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import final

import numpy as np

import syntalos_mlink as syl
from syntalos_mlink import DataType

GAZE_WORN_COL = 2
GAZE_GLOBAL_X_COL = 0
GAZE_GLOBAL_Y_COL = 1
GAZE_LEFT_X_COL = 23
GAZE_LEFT_Y_COL = 24
GAZE_RIGHT_X_COL = 25
GAZE_RIGHT_Y_COL = 26
GAZE_REQUIRED_COLS = 27

FRAME_WAIT_MAX_US = 500_000
FRESH_GAZE_MAX_DELTA_US = 20_000
GAZE_HISTORY_US = 2_000_000

CIRCLE_RADIUS_PX = 30
CIRCLE_THICKNESS_PX = 5

BGR_GLOBAL = (0, 0, 255)
BGR_LEFT = (0, 255, 0)
BGR_RIGHT = (255, 0, 0)
STALE_DIM_FACTOR = 0.35


@dataclass(frozen=True)
class GazeSample:
    timestamp_us: int
    worn: bool
    global_xy: tuple[float, float]
    left_xy: tuple[float, float]
    right_xy: tuple[float, float]


def dim_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(channel * STALE_DIM_FACTOR) for channel in color)  # pyright: ignore[reportReturnType]


def draw_circle_outline(
    mat: np.ndarray,
    xy: tuple[float, float],
    color: tuple[int, int, int],
    radius: int = CIRCLE_RADIUS_PX,
    thickness: int = CIRCLE_THICKNESS_PX,
) -> None:
    x_float, y_float = xy
    if not math.isfinite(x_float) or not math.isfinite(y_float):
        return

    height, width = mat.shape[:2]
    x = int(round(x_float))
    y = int(round(y_float))
    outer = radius
    inner = max(radius - thickness, 0)

    x0 = max(x - outer, 0)
    x1 = min(x + outer + 1, width)
    y0 = max(y - outer, 0)
    y1 = min(y + outer + 1, height)
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    mask = (dist2 <= outer**2) & (dist2 >= inner**2)
    if mat.ndim == 2:
        mat[y0:y1, x0:x1][mask] = max(color)
    else:
        view = mat[y0:y1, x0:x1]
        view[mask, :3] = color


def draw_gaze_overlay(mat: np.ndarray, gaze: GazeSample | None, stale: bool) -> None:
    if gaze is None or not gaze.worn:
        return

    global_color = dim_color(BGR_GLOBAL) if stale else BGR_GLOBAL
    left_color = dim_color(BGR_LEFT) if stale else BGR_LEFT
    right_color = dim_color(BGR_RIGHT) if stale else BGR_RIGHT

    draw_circle_outline(mat, gaze.global_xy, global_color)
    draw_circle_outline(mat, gaze.left_xy, left_color)
    draw_circle_outline(mat, gaze.right_xy, right_color)


# # ####################################################################################
# # Syntalos interface
# # ####################################################################################


@final
class Module:
    def __init__(self, mlink: syl.SyntalosLink) -> None:
        self.mlink = mlink
        self.register_ports()
        self.running: bool = False
        self.pending_frames: deque[syl.Frame] = deque()
        self.gaze_times: list[int] = []
        self.gaze_samples: list[GazeSample] = []
        self.last_gaze: GazeSample | None = None

    def register_ports(self) -> None:
        self.in_scene = self.mlink.register_input_port("scene", "Scene", data_type=DataType.Frame)
        self.in_gaze = self.mlink.register_input_port("gaze", "Gaze", DataType.FloatSignalBlock)
        self.out_scene = self.mlink.register_output_port("scene", "Scene", DataType.Frame)
        self.in_scene.on_data = self.on_scene
        self.in_gaze.on_data = self.on_gaze

    def prepare(self) -> bool:
        self.clear_buffers()

        framerate = self.in_scene.metadata.get("framerate", None)
        if framerate is not None:
            self.out_scene.set_metadata_value("framerate", framerate)

        frame_size = self.in_scene.metadata.get("size", None)
        if frame_size is not None:
            self.out_scene.set_metadata_value_size("size", frame_size)

        return True

    def start(self) -> None:
        self.running = True

    def event_loop_tick(self) -> None:
        if self.running:
            self.process_pending_frames()

    def stop(self) -> None:
        self.running = False
        self.clear_buffers()

    def clear_buffers(self) -> None:
        self.pending_frames.clear()
        self.gaze_times.clear()
        self.gaze_samples.clear()
        self.last_gaze = None

    def on_scene(self, frame: syl.Frame | None) -> None:
        if not self.running:
            return

        if frame is None:
            self.flush_pending_frames()
            return

        self.pending_frames.append(frame)
        self.process_pending_frames()

    def on_gaze(self, block: syl.FloatSignalBlock | None) -> None:
        if not self.running or block is None:
            return

        data = block.data
        timestamps = block.timestamps
        if data.ndim != 2 or data.shape[1] < GAZE_REQUIRED_COLS:
            raise ValueError(
                f"Expected gaze block with at least {GAZE_REQUIRED_COLS} columns, "
                + f"got shape {data.shape}"
            )

        for row_idx in range(min(len(timestamps), data.shape[0])):
            row = data[row_idx]
            sample = GazeSample(
                timestamp_us=int(timestamps[row_idx]),
                worn=bool(row[GAZE_WORN_COL] >= 0.5),
                global_xy=(float(row[GAZE_GLOBAL_X_COL]), float(row[GAZE_GLOBAL_Y_COL])),
                left_xy=(float(row[GAZE_LEFT_X_COL]), float(row[GAZE_LEFT_Y_COL])),
                right_xy=(float(row[GAZE_RIGHT_X_COL]), float(row[GAZE_RIGHT_Y_COL])),
            )
            self.gaze_times.append(sample.timestamp_us)
            self.gaze_samples.append(sample)
            self.last_gaze = sample

        self.process_pending_frames()

    def process_pending_frames(self) -> None:
        while self.pending_frames:
            frame = self.pending_frames[0]
            if not self.can_resolve_frame(int(frame.time_usec)):
                break

            _ = self.pending_frames.popleft()
            self.submit_frame_with_overlay(frame)

    def flush_pending_frames(self) -> None:
        while self.pending_frames:
            self.submit_frame_with_overlay(self.pending_frames.popleft())

    def can_resolve_frame(self, frame_time_us: int) -> bool:
        if self.gaze_times and self.gaze_times[-1] >= frame_time_us:
            return True

        now_us = int(syl.time_since_start_usec())

        return now_us >= frame_time_us + FRAME_WAIT_MAX_US

    def nearest_gaze(self, frame_time_us: int) -> tuple[GazeSample | None, bool]:
        if not self.gaze_times:
            return self.last_gaze, True

        insert_idx = bisect.bisect_left(self.gaze_times, frame_time_us)
        candidate_indexes: list[int] = []
        if insert_idx > 0:
            candidate_indexes.append(insert_idx - 1)
        if insert_idx < len(self.gaze_times):
            candidate_indexes.append(insert_idx)

        if not candidate_indexes:
            return self.last_gaze, True

        best_idx = min(
            candidate_indexes,
            key=lambda idx: abs(self.gaze_times[idx] - frame_time_us),
        )
        gaze = self.gaze_samples[best_idx]
        stale = abs(gaze.timestamp_us - frame_time_us) > FRESH_GAZE_MAX_DELTA_US
        return gaze, stale

    def prune_gaze_history(self, frame_time_us: int) -> None:
        cutoff = frame_time_us - GAZE_HISTORY_US
        keep_from = bisect.bisect_left(self.gaze_times, cutoff)
        if keep_from <= 0:
            return
        del self.gaze_times[:keep_from]
        del self.gaze_samples[:keep_from]

    def submit_frame_with_overlay(self, frame: syl.Frame) -> None:
        frame_time_us = int(frame.time_usec)
        gaze, stale = self.nearest_gaze(frame_time_us)
        draw_gaze_overlay(frame.mat, gaze, stale)
        self.out_scene.submit(frame)
        self.prune_gaze_history(frame_time_us)


def main() -> int:
    mlink = syl.init_link(rename_process=True)
    mod = Module(mlink)
    mlink.on_prepare = mod.prepare
    mlink.on_start = mod.start
    mlink.on_stop = mod.stop
    mlink.await_data_forever(mod.event_loop_tick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
