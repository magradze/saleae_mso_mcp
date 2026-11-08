from __future__ import annotations

import numpy as np
import pytest

from saleae_mso_mcp.analysis import decimate_minmax, measure_analog, measure_digital
from saleae_mso_mcp.domain import AnalogTrace, DigitalTrace

RATE = 1e6


def sine(freq: float, amp: float = 1.0, dur: float = 0.01, offset: float = 0.0) -> AnalogTrace:
    t = np.arange(int(dur * RATE)) / RATE
    return AnalogTrace(0.0, RATE, (offset + amp * np.sin(2 * np.pi * freq * t)).astype(np.float32))


def test_sine_measurements():
    m = measure_analog(sine(1000, amp=2.0, offset=0.5))
    assert m.frequency_hz == pytest.approx(1000, rel=1e-3)
    assert m.peak_to_peak_v == pytest.approx(4.0, rel=1e-3)
    assert m.mean_v == pytest.approx(0.5, abs=1e-3)
    assert m.ac_rms_v == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)
    assert m.duty_cycle == pytest.approx(0.5, abs=0.01)


def test_square_duty_and_edges():
    t = np.arange(10_000) / RATE
    v = np.where((t * 2000) % 1 < 0.25, 3.3, 0.0).astype(np.float32)
    m = measure_analog(AnalogTrace(0.0, RATE, v))
    assert m.frequency_hz == pytest.approx(2000, rel=1e-3)
    assert m.duty_cycle == pytest.approx(0.25, abs=0.01)
    assert m.rise_time_s is not None and m.rise_time_s <= 2 / RATE


def test_flat_signal_has_no_frequency():
    m = measure_analog(AnalogTrace(0.0, RATE, np.full(1000, 1.2, np.float32)))
    assert m.frequency_hz is None and m.peak_to_peak_v == pytest.approx(0)


def test_noise_does_not_create_false_edges():
    rng = np.random.default_rng(0)
    tr = sine(500)
    tr.voltages = tr.voltages + rng.normal(0, 0.05, tr.voltages.size).astype(np.float32)
    assert measure_analog(tr).frequency_hz == pytest.approx(500, rel=1e-2)


def test_pure_noise_has_no_frequency():
    rng = np.random.default_rng(0)
    v = rng.normal(0.02, 0.009, 100_000).astype(np.float32)
    m = measure_analog(AnalogTrace(0.0, RATE, v))
    assert m.frequency_hz is None and m.duty_cycle is None
    assert m.rise_time_s is None and m.fall_time_s is None
    assert m.peak_to_peak_v > 0


def test_decimation_keeps_spike():
    v = np.zeros(1_000_000, np.float32)
    v[123_457] = 5.0
    t, d = decimate_minmax(AnalogTrace(0.0, RATE, v), 200)
    assert len(d) <= 200
    assert d.max() == 5.0
    assert t[np.argmax(d)] == pytest.approx(123_457 / RATE)
    assert np.all(np.diff(t) >= 0)


def test_decimation_passthrough_when_small():
    t, d = decimate_minmax(AnalogTrace(1.0, RATE, np.arange(10, dtype=np.float32)), 100)
    assert len(d) == 10 and t[0] == 1.0


def test_digital_measurements():
    period = 1e-4
    edges = np.arange(1, 200) * (period / 2)
    m = measure_digital(DigitalTrace(0.0, 0.01, False, edges))
    assert m.frequency_hz == pytest.approx(10_000, rel=1e-6)
    assert m.duty_cycle == pytest.approx(0.5, abs=0.01)
    assert m.rising_edges + m.falling_edges == m.transitions == 199
    assert m.min_pulse_s == pytest.approx(period / 2)


def test_digital_window_state():
    tr = DigitalTrace(0.0, 1.0, False, np.array([0.1, 0.2, 0.3]))
    assert tr.state_at(0.15) is True
    w = tr.window(0.15, 0.25)
    assert w.initial_state is True and list(w.transition_times) == [0.2]


def test_analog_window():
    tr = AnalogTrace(0.0, 10.0, np.arange(10, dtype=np.float32))
    w = tr.window(0.2, 0.5)
    assert list(w.voltages) == [2, 3, 4, 5] and w.begin_time == pytest.approx(0.2)
