"""Runtime configuration, read from environment variables and CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path


class BackendKind(StrEnum):
    HARDWARE = "hardware"
    SIMULATED = "simulated"


def _default_capture_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "saleae-mso-mcp" / "captures"


@dataclass(frozen=True, slots=True)
class Settings:
    backend: BackendKind = BackendKind.HARDWARE
    serial_number: str | None = None
    capture_dir: Path = field(default_factory=_default_capture_dir)
    capture_timeout_s: float = 60.0
    max_waveform_points: int = 2000
    max_transitions: int = 1000
    # Highest analog rate of the Logic MSO; an omitted sample rate starts here.
    max_sample_rate_hz: float = 1.6e9
    # Per-channel sample budget used to divide the rate down for long captures.
    auto_max_samples: int = 20_000_000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if env is None else env)
        s = cls()
        if v := env.get("SALEAE_MSO_BACKEND"):
            s = replace(s, backend=BackendKind(v.lower()))
        if v := env.get("SALEAE_MSO_SERIAL"):
            s = replace(s, serial_number=v)
        if v := env.get("SALEAE_MSO_CAPTURE_DIR"):
            s = replace(s, capture_dir=Path(v).expanduser())
        if v := env.get("SALEAE_MSO_CAPTURE_TIMEOUT"):
            s = replace(s, capture_timeout_s=float(v))
        if v := env.get("SALEAE_MSO_MAX_POINTS"):
            s = replace(s, max_waveform_points=int(v))
        if v := env.get("SALEAE_MSO_MAX_TRANSITIONS"):
            s = replace(s, max_transitions=int(v))
        if v := env.get("SALEAE_MSO_MAX_SAMPLE_RATE"):
            s = replace(s, max_sample_rate_hz=float(v))
        if v := env.get("SALEAE_MSO_AUTO_MAX_SAMPLES"):
            s = replace(s, auto_max_samples=int(v))
        return s
