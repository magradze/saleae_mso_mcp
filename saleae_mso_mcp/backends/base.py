"""Backend contract. A backend owns the device and the on-disk capture format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from saleae_mso_mcp.domain import CaptureData, CaptureRequest


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    backend: str
    connected: bool
    serial_number: str | None = None
    part_number: str | None = None
    detail: str | None = None


@runtime_checkable
class Backend(Protocol):
    name: str

    def info(self) -> DeviceInfo:
        """Describe the device. Must not raise if the device is absent."""
        ...

    def capture(self, request: CaptureRequest, save_dir: Path, timeout_s: float) -> CaptureData:
        """Run a capture, persist it under ``save_dir`` and return its data."""
        ...

    def load(self, save_dir: Path, request: CaptureRequest) -> CaptureData:
        """Load a capture this backend previously wrote to ``save_dir``."""
        ...

    def close(self) -> None:
        """Release the device so other software (e.g. Logic 2) can use it."""
        ...
