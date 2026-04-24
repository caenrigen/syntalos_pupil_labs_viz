"""Pupil Labs Visualization Syntalos Module."""

from typing import final

import syntalos_mlink as syl
from syntalos_mlink import DataType


# # ####################################################################################
# # Syntalos interface
# # ####################################################################################


@final
class Module:
    def __init__(self, mlink: syl.SyntalosLink) -> None:
        self.mlink = mlink
        self.register_ports()
        self.running: bool = False

    def register_ports(self) -> None:
        self.in_scene = self.mlink.register_input_port("scene", "Scene", data_type=DataType.Frame)
        self.in_gaze = self.mlink.register_input_port("gaze", "Gaze", DataType.FloatSignalBlock)
        self.out_scene = self.mlink.register_output_port("scene", "Scene", DataType.Frame)

    def prepare(self):
        return True

    def start(self) -> None:
        self.running = True
        pass

    def event_loop_tick(self) -> None:
        pass

    def stop(self) -> None:
        self.running = False
        pass


def main() -> int:
    mlink = syl.init_link()
    mod = Module(mlink)
    mlink.on_prepare = mod.prepare
    mlink.on_start = mod.start
    mlink.on_stop = mod.stop
    mlink.await_data_forever(mod.event_loop_tick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
