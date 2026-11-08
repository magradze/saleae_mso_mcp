from __future__ import annotations

import pytest

from saleae_mso_mcp.backends.simulated import SimulatedBackend
from saleae_mso_mcp.domain import (
    AnalogChannelSpec,
    CaptureRequest,
    DigitalChannelSpec,
    EdgeTriggerSpec,
    TimedMode,
    TriggeredMode,
)
from saleae_mso_mcp.errors import CaptureFailedError, NotFoundError
from saleae_mso_mcp.service import MsoService


def timed(**kw) -> CaptureRequest:
    return CaptureRequest(
        analog=[AnalogChannelSpec(channel=0)],
        digital=[DigitalChannelSpec(port=0, channel=0)],
        sample_rate_hz=1e6,
        mode=TimedMode(duration_s=0.01),
        **kw,
    )


def test_timed_capture_summary(service):
    s = service.capture(timed(label="bench"))
    ch1, d0 = s["channels"]
    assert ch1["name"] == "CH1" and ch1["measurements"]["frequency_hz"] == pytest.approx(1000, rel=1e-3)
    assert d0["name"] == "D0.0" and d0["measurements"]["frequency_hz"] == pytest.approx(10_000, rel=1e-3)
    assert s["label"] == "bench"


def test_captures_persist_and_reload(service, settings):
    cid = service.capture(timed())["id"]
    fresh = MsoService(settings, SimulatedBackend())  # empty cache: must load from disk
    assert fresh.summary(cid)["channels"][0]["measurements"]["frequency_hz"] == pytest.approx(1000, rel=1e-3)
    assert [c["id"] for c in fresh.list_captures()["captures"]] == [cid]


def test_measure_window_and_waveform(service):
    cid = service.capture(timed())["id"]
    m = service.measure(cid, "CH1", 0.002, 0.004)
    assert m["duration_s"] == pytest.approx(0.002, rel=1e-2)
    w = service.waveform(cid, "CH1", None, None, 100)
    seg = w["segments"][0]
    assert seg["decimated"] and len(seg["points"]) <= 100


def test_transitions_and_csv(service):
    cid = service.capture(timed())["id"]
    tr = service.transitions(cid, "D0.0", 0, 0.001, 5)
    assert tr["segments"][0]["truncated"] and len(tr["segments"][0]["times_s"]) == 5
    out = service.export_csv(cid, "CH1", 0, 0.0001)
    assert out["rows"] == 101


def test_triggered_capture_aligns_edge(service):
    req = CaptureRequest(
        analog=[AnalogChannelSpec(channel=1)],
        sample_rate_hz=10e6,
        mode=TriggeredMode(
            trigger=EdgeTriggerSpec(channel="CH2", threshold_volts=0.5), pre_trigger_s=1e-4, post_trigger_s=1e-4
        ),
    )
    cid = service.capture(req)["id"]
    before = service.measure(cid, "CH2", -2e-6, -1e-6)["mean_v"]
    after = service.measure(cid, "CH2", 1e-6, 2e-6)["mean_v"]
    assert before < 0.5 < after


def test_trigger_timeout_cleans_up(service):
    req = CaptureRequest(
        analog=[AnalogChannelSpec(channel=0)],
        mode=TriggeredMode(trigger=EdgeTriggerSpec(channel="CH1", threshold_volts=5.0)),
    )
    with pytest.raises(CaptureFailedError):
        service.capture(req)
    assert service.list_captures()["captures"] == []


def test_errors(service):
    cid = service.capture(timed())["id"]
    with pytest.raises(NotFoundError):
        service.measure(cid, "nope", None, None)
    with pytest.raises(NotFoundError):
        service.summary("../etc")
    service.delete(cid)
    with pytest.raises(NotFoundError):
        service.summary(cid)


def test_omitted_rate_uses_device_maximum(service, settings):
    req = CaptureRequest(analog=[AnalogChannelSpec(channel=0)], mode=TimedMode(duration_s=1e-4))
    s = service.capture(req)
    assert s["requested_sample_rate_hz"] == settings.max_sample_rate_hz
    assert s["sample_rate_hz"] == pytest.approx(settings.max_sample_rate_hz)
    assert "sample_rate_note" not in s


def test_omitted_rate_divides_down_for_long_captures(service, settings):
    req = CaptureRequest(analog=[AnalogChannelSpec(channel=0)], mode=TimedMode(duration_s=0.05))
    rate = service._resolve_rate(req).sample_rate_hz
    assert rate * 0.05 <= settings.auto_max_samples
    assert settings.max_sample_rate_hz / rate == pytest.approx(round(settings.max_sample_rate_hz / rate))


def test_request_validation():
    with pytest.raises(ValueError):
        CaptureRequest(mode=TimedMode(duration_s=1))
    with pytest.raises(ValueError):
        CaptureRequest(
            analog=[AnalogChannelSpec(channel=0)], mode=TriggeredMode(trigger=EdgeTriggerSpec(channel="CH9"))
        )
