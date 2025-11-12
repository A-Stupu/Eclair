"""Arduino turntable controller helper.

Provides a thin wrapper around pyserial for common operations and a
local simulator when serial isn't available. The interface is intentionally
small: connect(), disconnect(), rotate_degrees(), step(), home().
"""
from __future__ import annotations

import logging
import time
from typing import Optional

try:
    import serial
except Exception:  # pragma: no cover - serial may not be installed in CI
    serial = None  # type: ignore


LOG = logging.getLogger(__name__)


class ArduinoController:
    def __init__(self, port: Optional[str] = None, baud: int = 115200, step_delay: float = 0.3) -> None:
        self.port = port
        self.baud = baud
        self.step_delay = step_delay
        self._ser = None
        self._simulated = serial is None or port is None

    def connect(self) -> bool:
        if self._simulated:
            LOG.info("ArduinoController: running in simulated mode")
            return True
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(2)
            LOG.info("Connected to Arduino on %s", self.port)
            return True
        except Exception as exc:
            LOG.exception("Failed to open serial port %s: %s", self.port, exc)
            self._ser = None
            self._simulated = True
            return False

    def disconnect(self) -> None:
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                LOG.exception("Error while closing serial")
            self._ser = None

    def _write(self, data: str) -> None:
        if self._simulated:
            LOG.debug("[SIM] Write to Arduino: %s", data)
            return
        if not self._ser:
            raise RuntimeError("Arduino not connected")
        self._ser.write(data.encode("utf-8") + b"\n")

    def rotate_degrees(self, degrees: float) -> None:
        """Rotate the turntable by degrees (positive clockwise).

        The serial protocol is intentionally simple: send a line like
        "ROTATE 10" to rotate 10 degrees. The Arduino firmware should
        accept and handle that command. If no hardware is present, we
        simply log and sleep a short amount to simulate motion.
        """
        cmd = f"ROTATE {degrees}"
        try:
            self._write(cmd)
        except Exception:
            LOG.exception("Failed to send rotate command")
        # small delay to let physical turntable move; adjustable via step_delay
        time.sleep(self.step_delay)

    def step(self, steps: int = 1, degrees_per_step: float = 10.0) -> None:
        self.rotate_degrees(steps * degrees_per_step)

    def home(self) -> None:
        """Return turntable to home position. Protocol: send HOME."""
        try:
            self._write("HOME")
        except Exception:
            LOG.exception("Failed to send home command")
        time.sleep(self.step_delay)


def make_controller(port: Optional[str] = None, baud: int = 115200) -> ArduinoController:
    return ArduinoController(port=port, baud=baud)
