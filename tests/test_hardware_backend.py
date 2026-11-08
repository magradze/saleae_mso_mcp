"""Hardware backend against a stand-in for ``saleae.mso_api`` (no device needed)."""

from __future__ import annotations

import sys
import types
from enum import Enum
from typing import Any

import numpy as np
import pytest

from saleae_mso_mcp.domain import (
    AnalogChannelSpec,
    CaptureRequest,
    DigitalChannelSpec,
    EdgeTriggerSpec,
    TimedMode,
    TriggeredMode,
)
from saleae_mso_mcp.errors import CaptureFailedError


def _fake_api() -> types.ModuleType:
    api = types.ModuleType("saleae.mso_api")

    class ProbeAttenuation(Enum):
        PROBE_1X = 1
        PROBE_10X = 10

    class Coupling(Enum):
        DC = 0
        AC = 1

    class BandwidthLimit(Enum):
        MHz_20 = 20
        MHz_350 = 350

    class DownsampleMode(Enum):
        DROP = 0
        AVERAGE = 1

    class EdgeTriggerDirection(Enum):
        RISING = 0
        FALLING = 1

    class _Rec:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    for n in [
        "AnalogChannel",
        "DigitalChannel",
        "AnalogSettings",
        "TimedCapture",
        "AnalogTriggered",
        "EdgeTrigger",
        "CaptureConfig",
    ]:
        setattr(api, n, type(n, (_Rec,), {}))

    class CaptureTimeoutError(Exception):
        pass

    class AnalogData(_Rec):
        pass

    class DigitalData(_Rec):
        pass

    class Capture(_Rec):
        pass

    class MSO:
        last_config: Any = None

        def __init__(self, serial_number: str | None = None) -> None:
            self.part_number = types.SimpleNamespace(name="MSO_4CH_200MHZ")
            self.closed = False

        def capture(self, config: Any, save_dir: Any, timeout_secs: float | None = None) -> Any:
            MSO.last_config = config
            if (
                isinstance(config.capture_settings, api.AnalogTriggered)
                and config.capture_settings.trigger.threshold_volts > 100
            ):
                raise CaptureTimeoutError("no trigger")
            n = 1000
            t = np.arange(n) / 1e6
            analog = {
                c.name: AnalogData(
                    voltages=np.sin(2 * np.pi * 5000 * t).astype(np.float32),
                    begin_time=0.0,
                    end_time=n / 1e6,
                    sample_rate=1.6e9,
                    downsample=1600,
                )
                for c in config.enabled_channels
                if isinstance(c, api.AnalogChannel)
            }
            digital = {
                c.name: DigitalData(initial_state=False, transition_times=np.arange(1, 20) * 5e-5)
                for c in config.enabled_channels
                if isinstance(c, api.DigitalChannel)
            }
            return Capture(analog_data=analog, digital_data=digital, analog_start_time=0.0, analog_stop_time=n / 1e6)

        def close(self) -> None:
            self.closed = True

    for obj in [ProbeAttenuation, Coupling, BandwidthLimit, DownsampleMode, EdgeTriggerDirection, MSO, Capture]:
        setattr(api, obj.__name__, obj)
    api.CaptureTimeoutError = CaptureTimeoutError
    return api


@pytest.fixture
def api(monkeypatch):
    fake = _fake_api()
    pkg = types.ModuleType("saleae")
    pkg.mso_api = fake
    monkeypatch.setitem(sys.modules, "saleae", pkg)
    monkeypatch.setitem(sys.modules, "saleae.mso_api", fake)
    return fake


def test_config_mapping_and_conversion(api, tmp_path):
    from saleae_mso_mcp.backends.hardware import HardwareBackend

    be = HardwareBackend()
    req = CaptureRequest(
        analog=[AnalogChannelSpec(channel=0, name="vin", probe="1x", coupling="ac", bandwidth="20mhz")],
        digital=[DigitalChannelSpec(port=0, channel=3, threshold_volts=1.6)],
        sample_rate_hz=1e6,
        mode=TimedMode(duration_s=0.001),
    )
    data = be.capture(req, tmp_path, 5)

    cfg = api.MSO.last_config
    a, d = cfg.enabled_channels
    assert a.probe_attenuation is api.ProbeAttenuation.PROBE_1X
    assert a.coupling is api.Coupling.AC and a.bandwidth_mhz is api.BandwidthLimit.MHz_20
    assert (d.port, d.channel, d.threshold_volts) == (0, 3, 1.6)
    assert cfg.capture_settings.capture_length_seconds == 0.001
    assert cfg.analog_settings.downsample_mode is api.DownsampleMode.AVERAGE

    tr = data.analog["vin"].traces[0]
    assert tr.sample_rate == pytest.approx(1e6)
    assert data.digital["D0.3"].traces[0].transition_times.size == 19
    assert be.info().part_number == "MSO_4CH_200MHZ"


def test_trigger_mapping_and_error_translation(api, tmp_path):
    from saleae_mso_mcp.backends.hardware import HardwareBackend

    be = HardwareBackend()
    trig = EdgeTriggerSpec(channel="CH1", threshold_volts=1.0, direction="falling")
    req = CaptureRequest(analog=[AnalogChannelSpec(channel=0)], mode=TriggeredMode(trigger=trig))
    be.capture(req, tmp_path, 5)
    st = api.MSO.last_config.capture_settings
    assert st.trigger.channel_name == "CH1" and st.trigger.direction is api.EdgeTriggerDirection.FALLING

    bad = CaptureRequest(
        analog=[AnalogChannelSpec(channel=0)],
        mode=TriggeredMode(trigger=EdgeTriggerSpec(channel="CH1", threshold_volts=500)),
    )
    with pytest.raises(CaptureFailedError, match="No trigger was found"):
        be.capture(bad, tmp_path, 5)
