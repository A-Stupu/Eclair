"""Runtime controller for the Arduino-driven turntable.

This module provides a small class-based wrapper so the GUI can interact with
the hardware without modifying the legacy ``arduino_controller.py`` script.
"""
from __future__ import annotations

import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

try:  # pragma: no cover - optional dependency handling
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    SERIAL_IMPORT_ERROR = exc
    serial = None  # type: ignore
else:
    SERIAL_IMPORT_ERROR = None

__all__ = ["TurntableController", "TurntableError"]

_PORT_KEYWORDS = ("arduino", "ch340", "ttyusb", "ttyacm", "cp210")


class TurntableError(RuntimeError):
    """Raised when the turntable cannot be controlled."""


@dataclass
class TurntableController:
	baudrate: int = 9600
	step_size: float = 20.0
	timeout: float = 1.0
	port_hint: Optional[str] = None

	_serial: Optional[Any] = None
	_port: Optional[str] = None
	_position: float = 0.0
	_env_port: Optional[str] = None

	def __post_init__(self) -> None:
		self._env_port = os.getenv("ECLAIR_TURNTABLE_PORT")
		self._ensure_support()
		self.refresh_port()

	def _ensure_support(self) -> None:
		if SERIAL_IMPORT_ERROR is not None:
			raise TurntableError("pyserial is not installed.") from SERIAL_IMPORT_ERROR
		if serial is None:  # pragma: no cover - defensive guard
			raise TurntableError("pyserial is not installed.")

	@property
	def is_available(self) -> bool:
		return self._port is not None

	@property
	def is_connected(self) -> bool:
		return self._serial is not None and self._serial.is_open

	def refresh_port(self) -> bool:
		self._ensure_support()
		port = self._find_port()
		self._port = port
		return port is not None

	def connect(self) -> None:
		self._ensure_support()
		if not self._port and not self.refresh_port():
			raise TurntableError("Unable to detect an Arduino turntable on the serial ports.")

		if self.is_connected:
			return

		try:
			self._serial = serial.Serial(self._port, baudrate=self.baudrate, timeout=self.timeout)
		except Exception as exc:  # pragma: no cover - depends on hardware availability
			raise TurntableError(f"Failed to open serial port {self._port}: {exc}") from exc

		time.sleep(2)  # allow Arduino reset after opening the port
		self._position = 0.0

	def disconnect(self) -> None:
		if self._serial and self._serial.is_open:
			try:
				self._serial.close()
			except Exception:
				pass
		self._serial = None

	def rotate_degrees(self, degrees: float) -> None:
		if not self.is_connected:
			raise TurntableError("Turntable is not connected.")

		target = self._position + degrees
		target = max(0.0, min(180.0, target))

		# if self.step_size > 0:
		# 	target = round(target / self.step_size) * self.step_size
		# 	target = max(0.0, min(180.0, target))

		# command = f"{int(target)}\n".encode("ascii")
		command = f"{int(target)}\n".encode("utf-8")
		# command = (str(target) + '\n').encode('utf-8')
  
		try:
			self._serial.write(command)
			self._serial.flush()
		except Exception as exc:  # pragma: no cover - depends on hardware availability
			raise TurntableError(f"Failed to send command to turntable: {exc}") from exc

		time.sleep(0.15)
		self._position = target

	def home(self) -> None:
		if not self.is_connected:
			return
		if self._position:
			self.rotate_degrees(-self._position)

	def list_available_ports(self) -> List[str]:
		return [port.device for port in serial.tools.list_ports.comports() if port.device]

	def _find_port(self) -> Optional[str]:
		ports = list(serial.tools.list_ports.comports())
		preferred = self.port_hint or self._env_port
		if preferred:
			preferred = preferred.strip()
			if preferred and self._port_is_present(preferred, ports):
				return preferred
			path = Path(preferred)
			if path.exists():
				return preferred

		for port in ports:
			description = (port.description or "").lower()
			if any(key in description for key in _PORT_KEYWORDS):
				return port.device
			device = (port.device or "").lower()
			if any(key in device for key in _PORT_KEYWORDS):
				return port.device

		return ports[0].device if ports else None

	@staticmethod
	def _port_is_present(target: str, ports: List[Any]) -> bool:
		target = target.strip()
		for port in ports:
			if not port.device:
				continue
			if port.device == target or port.name == target:
				return True
		return False

	def __del__(self) -> None:  # pragma: no cover - defensive cleanup
		try:
			self.disconnect()
		except Exception:
			pass
