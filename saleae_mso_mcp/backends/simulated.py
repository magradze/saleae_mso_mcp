"""Synthetic backend for development, CI and demos without hardware.

Analog channel n carries a (n+1) kHz sine (CH1 1 kHz, CH2 2 kHz ...) of 1.5 V
amplitude plus a little noise. Digital channel (port p, bit b) carries a square
wave at (b+1) * 10 kHz, phase-shifted by port.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from saleae_mso_mcp.backends.base import DeviceInfo
from saleae_mso_mcp.domain import (
    AnalogChannelData,
    AnalogTrace,
    CaptureData,
    CaptureRequest,
    DigitalChannelData,
    DigitalTrace,
    EdgeDirection,
    TimedMode,
)
from saleae_mso_mcp.errors import CaptureFailedError, InvalidRequestError

MAX_SAMPLES = 20_000_000
AMPLITUDE = 1.5
NOISE = 0.01


def _sine_freq(channel: int) -> float:
    return 1_000.0 * (channel + 1)


def _square_freq(bit: int) -> float:
    return 10_000.0 * (bit + 1)


class SimulatedBackend:
    name = "simulated"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def info(self) -> DeviceInfo:
        return DeviceInfo(self.name, True, "SIM-0001", "simulated", "synthetic signals, no hardware")

    def close(self) -> None:
        pass

    # -- capture -----------------------------------------------------------

    def capture(self, request: CaptureRequest, save_dir: Path, timeout_s: float) -> CaptureData:
        mode = request.mode
        if isinstance(mode, TimedMode):
            t0, t1, phase0 = 0.0, mode.duration_s, 0.0
        else:
            spec = next(c for c in request.analog if c.label == mode.trigger.channel)
            level = mode.trigger.threshold_volts
            if abs(level) >= AMPLITUDE:
                raise CaptureFailedError(
                    f"no trigger within {mode.timeout_s} s: threshold {level} V is outside the "
                    f"signal's ±{AMPLITUDE} V swing"
                )
            # Shift the trigger channel's phase so its edge crosses the level at t = 0.
            base = math.asin(level / AMPLITUDE)
            phase0 = base if mode.trigger.direction == EdgeDirection.RISING else math.pi - base
            phase0 /= 2 * math.pi * _sine_freq(spec.channel)
            t0, t1 = -mode.pre_trigger_s, mode.post_trigger_s

        rate = request.sample_rate_hz
        n = round((t1 - t0) * rate)
        if n <= 0:
            raise InvalidRequestError("capture window shorter than one sample")
        if n > MAX_SAMPLES:
            raise InvalidRequestError(f"simulated capture limited to {MAX_SAMPLES} samples, got {n}")
        t = t0 + np.arange(n) / rate

        data = CaptureData()
        arrays: dict[str, np.ndarray] = {}
        for spec in request.analog:
            f = _sine_freq(spec.channel)
            v = AMPLITUDE * np.sin(2 * np.pi * f * (t + phase0))
            v = v + self._rng.normal(0, NOISE, n)
            lo = spec.center_voltage - spec.voltage_range / 2
            hi = spec.center_voltage + spec.voltage_range / 2
            v = np.clip(v, lo, hi).astype(np.float32)
            data.analog[spec.label] = AnalogChannelData(spec.label, spec.channel, [AnalogTrace(t0, rate, v)])
            arrays[f"a::{spec.label}"] = v

        for spec in request.digital:
            f = _square_freq(spec.channel)
            half = 0.5 / f
            shift = spec.port * half / 2
            first = math.ceil((t0 + shift) / half)
            edges = np.arange(first, math.floor((t1 + shift) / half) + 1) * half - shift
            edges = edges[(edges > t0) & (edges <= t1)]
            init = bool(math.floor((t0 + shift) / half) % 2 == 0)
            tr = DigitalTrace(t0, t1, init, edges)
            data.digital[spec.label] = DigitalChannelData(spec.label, spec.port, spec.channel, [tr])
            arrays[f"d::{spec.label}"] = edges
            arrays[f"i::{spec.label}"] = np.array([init])

        arrays["__window__"] = np.array([t0, t1, rate])
        np.savez_compressed(save_dir / "simulated.npz", **arrays)
        return data

    def load(self, save_dir: Path, request: CaptureRequest) -> CaptureData:
        z = np.load(save_dir / "simulated.npz")
        t0, t1, rate = (float(x) for x in z["__window__"])
        data = CaptureData()
        for spec in request.analog:
            v = z[f"a::{spec.label}"]
            data.analog[spec.label] = AnalogChannelData(spec.label, spec.channel, [AnalogTrace(t0, rate, v)])
        for spec in request.digital:
            tr = DigitalTrace(t0, t1, bool(z[f"i::{spec.label}"][0]), z[f"d::{spec.label}"])
            data.digital[spec.label] = DigitalChannelData(spec.label, spec.port, spec.channel, [tr])
        return data
