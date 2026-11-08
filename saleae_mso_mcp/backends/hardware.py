"""Logic MSO hardware backend, built on Saleae's ``saleae-mso-api`` package.

The vendor package is imported lazily so the rest of the server (and the
simulated backend) works without it installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from saleae_mso_mcp.backends.base import DeviceInfo
from saleae_mso_mcp.domain import (
    AnalogChannelData,
    AnalogTrace,
    BandwidthLimit,
    CaptureData,
    CaptureRequest,
    Coupling,
    DigitalChannelData,
    DigitalTrace,
    DownsampleMode,
    EdgeDirection,
    ProbeAttenuation,
    TimedMode,
)
from saleae_mso_mcp.errors import CaptureFailedError, DeviceUnavailableError

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "saleae-mso-api is not installed. Install it with: "
    "pip install 'saleae-mso-mcp[hardware]' --extra-index-url https://downloads.saleae.com/pypi/"
)

# Vendor exception class name -> advice appended to the message.
_ADVICE = {
    "USBPermissionsError": "USB permission denied. Install the udev rules: python -m saleae.mso_api.utils.install_udev",
    "SaleaeCommsProcessError": "Lost contact with the device. Check the USB cable and close "
    "Logic 2 if it is open (only one program can own the MSO).",
    "CaptureTimeoutError": "No trigger was found, or the capture took longer than the timeout. "
    "Check the trigger channel, level and direction.",
    "SmartCableNotReadyError": "The digital probe (smart cable) is missing or not ready.",
    "CaptureConfigError": "The device rejected the capture settings.",
}


def _mso_api() -> Any:
    try:
        from saleae import mso_api
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DeviceUnavailableError(INSTALL_HINT) from exc
    return mso_api


def _translate(exc: Exception) -> Exception:
    name = type(exc).__name__
    advice = _ADVICE.get(name)
    msg = f"{name}: {exc}".strip()
    if advice:
        msg = f"{msg}. {advice}"
    return CaptureFailedError(msg)


class HardwareBackend:
    name = "hardware"

    def __init__(self, serial_number: str | None = None) -> None:
        self._serial = serial_number
        self._mso: Any | None = None

    # -- device lifecycle --------------------------------------------------

    def _device(self) -> Any:
        if self._mso is None:
            api = _mso_api()
            try:
                self._mso = api.MSO(serial_number=self._serial)
            except Exception as exc:
                raise DeviceUnavailableError(
                    f"Could not open Logic MSO ({type(exc).__name__}: {exc}). Is it plugged in, and is Logic 2 closed?"
                ) from exc
        return self._mso

    def info(self) -> DeviceInfo:
        try:
            mso = self._device()
        except DeviceUnavailableError as exc:
            return DeviceInfo(self.name, False, self._serial, None, str(exc))
        part = getattr(mso, "part_number", None)
        serial = self._serial or getattr(getattr(mso, "wrapper", None), "serial_number", None)
        return DeviceInfo(self.name, True, serial, getattr(part, "name", None) or str(part))

    def close(self) -> None:
        if self._mso is not None:
            try:
                self._mso.close()
            except Exception:  # pragma: no cover - best effort
                log.exception("error closing MSO")
            self._mso = None

    # -- capture -----------------------------------------------------------

    def _config(self, request: CaptureRequest) -> Any:
        api = _mso_api()
        # Coupling and BandwidthLimit are not re-exported from saleae.mso_api.
        from saleae.mso_api import capture_config as cc

        probe = {
            ProbeAttenuation.X1: api.ProbeAttenuation.PROBE_1X,
            ProbeAttenuation.X10: api.ProbeAttenuation.PROBE_10X,
        }
        coupling = {Coupling.DC: cc.Coupling.DC, Coupling.AC: cc.Coupling.AC}
        bw = {BandwidthLimit.MHZ_20: cc.BandwidthLimit.MHz_20, BandwidthLimit.MHZ_350: cc.BandwidthLimit.MHz_350}
        ds = {DownsampleMode.AVERAGE: api.DownsampleMode.AVERAGE, DownsampleMode.DROP: api.DownsampleMode.DROP}

        channels: list[Any] = [
            api.AnalogChannel(
                channel=c.channel,
                name=c.label,
                center_voltage=c.center_voltage,
                voltage_range=c.voltage_range,
                probe_attenuation=probe[c.probe],
                coupling=coupling[c.coupling],
                bandwidth_mhz=bw[c.bandwidth],
            )
            for c in request.analog
        ]
        channels += [
            api.DigitalChannel(channel=c.channel, name=c.label, port=c.port, threshold_volts=c.threshold_volts)
            for c in request.digital
        ]

        mode = request.mode
        if isinstance(mode, TimedMode):
            settings = api.TimedCapture(capture_length_seconds=mode.duration_s)
        else:
            direction = (
                api.EdgeTriggerDirection.RISING
                if mode.trigger.direction == EdgeDirection.RISING
                else api.EdgeTriggerDirection.FALLING
            )
            settings = api.AnalogTriggered(
                trigger=api.EdgeTrigger(
                    channel_name=mode.trigger.channel,
                    threshold_volts=mode.trigger.threshold_volts,
                    direction=direction,
                    holdoff_seconds=mode.trigger.holdoff_s,
                ),
                pre_trigger_seconds=mode.pre_trigger_s,
                post_trigger_seconds=mode.post_trigger_s,
                timeout_seconds=mode.timeout_s,
            )

        return api.CaptureConfig(
            enabled_channels=channels,
            analog_settings=api.AnalogSettings(
                sample_rate=request.sample_rate_hz, downsample_mode=ds[request.downsample]
            ),
            capture_settings=settings,
        )

    def capture(self, request: CaptureRequest, save_dir: Path, timeout_s: float) -> CaptureData:
        mso = self._device()
        config = self._config(request)
        raw_dir = save_dir / "raw"
        try:
            cap = mso.capture(config, raw_dir, timeout_secs=timeout_s)
        except Exception as exc:
            if type(exc).__name__ == "SaleaeCommsProcessError":
                self.close()  # force a fresh connection next time
            raise _translate(exc) from exc
        return _convert(cap, request)

    def load(self, save_dir: Path, request: CaptureRequest) -> CaptureData:
        api = _mso_api()
        return _convert(api.Capture.from_dir(save_dir / "raw"), request)


# ---------------------------------------------------------------------------
# Vendor Capture -> domain
# ---------------------------------------------------------------------------


def _analog_trace(d: Any) -> AnalogTrace:
    v = np.asarray(d.voltages, dtype=np.float32)
    begin = float(getattr(d, "begin_time", 0.0))
    end = getattr(d, "end_time", None)
    rate = float(getattr(d, "sample_rate", 0.0)) / max(int(getattr(d, "downsample", 1) or 1), 1)
    if end is not None and len(v) > 1 and float(end) > begin:
        measured = len(v) / (float(end) - begin)
        # Trust the time base derived from begin/end if the nominal rate disagrees.
        if rate <= 0 or abs(measured - rate) / measured > 0.05:
            rate = measured
    if rate <= 0:
        raise CaptureFailedError("could not determine analog sample rate from capture")
    return AnalogTrace(begin, rate, v)


def _digital_trace(d: Any, begin: float, end: float) -> DigitalTrace:
    tt = np.asarray(d.transition_times, dtype=np.float64)
    b = float(getattr(d, "begin_time", begin))
    e = float(getattr(d, "end_time", end))
    return DigitalTrace(b, e, bool(d.initial_state), tt)


def _convert(cap: Any, request: CaptureRequest) -> CaptureData:
    out = CaptureData()
    try:
        begin, end = float(cap.analog_start_time), float(cap.analog_stop_time)
    except Exception:
        begin, end = 0.0, 0.0

    by_label = {c.label: c for c in request.analog}
    for name, d in cap.analog_data.items():
        traces = [_analog_trace(t) for t in getattr(d, "traces", [d])]
        spec = by_label.get(name)
        ch = spec.channel if spec else int(getattr(d, "channel", -1))
        out.analog[name] = AnalogChannelData(name, ch, traces)

    dig = {c.label: c for c in request.digital}
    for name, d in cap.digital_data.items():
        segments = getattr(d, "segments", [d])
        if end <= begin:
            ends = [float(s.transition_times[-1]) if len(s.transition_times) else 0.0 for s in segments]
            end = max([*ends, begin])
        traces = [_digital_trace(s, begin, end) for s in segments]
        spec = dig.get(name)
        out.digital[name] = DigitalChannelData(name, spec.port if spec else -1, spec.channel if spec else -1, traces)
    return out
