"""Signal measurements and data reduction.

Everything here is pure: numpy in, plain Python numbers out. The goal is to
give a language model compact, accurate facts about a waveform instead of
millions of raw samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from saleae_mso_mcp.domain import AnalogTrace, DigitalTrace


@dataclass(frozen=True, slots=True)
class AnalogMeasurements:
    samples: int
    duration_s: float
    min_v: float
    max_v: float
    mean_v: float
    rms_v: float
    ac_rms_v: float
    peak_to_peak_v: float
    frequency_hz: float | None
    period_s: float | None
    duty_cycle: float | None
    rise_time_s: float | None
    fall_time_s: float | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DigitalMeasurements:
    duration_s: float
    initial_state: int
    final_state: int
    transitions: int
    rising_edges: int
    falling_edges: int
    frequency_hz: float | None
    period_s: float | None
    duty_cycle: float | None
    min_pulse_s: float | None
    max_pulse_s: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _f(x: float | np.floating | None) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _crossings(v: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices of rising and falling crossings using a Schmitt trigger (hysteresis)."""
    state = np.full(v.shape, -1, dtype=np.int8)
    state[v >= hi] = 1
    state[v <= lo] = 0
    known = np.flatnonzero(state >= 0)
    if known.size < 2:
        return np.empty(0, int), np.empty(0, int)
    seq = state[known]
    change = np.flatnonzero(np.diff(seq) != 0) + 1
    idx = known[change]
    rising = idx[seq[change] == 1]
    falling = idx[seq[change] == 0]
    return rising, falling


# Rising-edge spacing more irregular than this means the "edges" are noise, not a periodic signal.
MAX_PERIOD_SPREAD = 0.25


def _irregular(rising: np.ndarray) -> bool:
    """True when rising edges are too unevenly spaced to describe a periodic signal."""
    if rising.size < 3:
        return False
    gaps = np.diff(rising).astype(np.float64)
    return float(gaps.std() / gaps.mean()) > MAX_PERIOD_SPREAD


def _edge_time(v: np.ndarray, dt: float, idx: int, a: float, b: float) -> float | None:
    """Time for the signal to go from level a to level b around a crossing index."""
    n = len(v)
    j = idx
    rising = b > a
    while j > 0 and ((v[j] > a) if rising else (v[j] < a)):
        j -= 1
    k = idx
    while k < n - 1 and ((v[k] < b) if rising else (v[k] > b)):
        k += 1
    if j == 0 or k == n - 1:
        return None
    return (k - j) * dt


def measure_analog(trace: AnalogTrace) -> AnalogMeasurements:
    v = np.asarray(trace.voltages, dtype=np.float64)
    n = int(v.size)
    dt = 1.0 / trace.sample_rate
    if n == 0:
        return AnalogMeasurements(0, 0.0, *([None] * 11))  # type: ignore[arg-type]

    vmin, vmax = float(v.min()), float(v.max())
    mean = float(v.mean())
    rms = float(np.sqrt(np.mean(v * v)))
    ac_rms = float(v.std())
    pp = vmax - vmin

    freq = period = duty = rise = fall = None
    if pp > 1e-9:
        mid = (vmin + vmax) / 2
        hyst = 0.1 * pp
        rising, falling = _crossings(v, mid - hyst, mid + hyst)
        if _irregular(rising):
            # Crossings of an open input or a noise floor: report levels only.
            rising, falling = rising[:0], falling[:0]
        if rising.size >= 2:
            # Use whole periods between first and last rising edge.
            first, last = int(rising[0]), int(rising[-1])
            period = (last - first) * dt / (rising.size - 1)
            freq = 1.0 / period
            duty = float(np.mean(v[first:last] > mid))
        lo10, hi90 = vmin + 0.1 * pp, vmin + 0.9 * pp
        rises = [t for i in rising[:50] if (t := _edge_time(v, dt, int(i), lo10, hi90))]
        falls = [t for i in falling[:50] if (t := _edge_time(v, dt, int(i), hi90, lo10))]
        rise = float(np.median(rises)) if rises else None
        fall = float(np.median(falls)) if falls else None

    return AnalogMeasurements(
        samples=n,
        duration_s=n * dt,
        min_v=vmin,
        max_v=vmax,
        mean_v=mean,
        rms_v=rms,
        ac_rms_v=ac_rms,
        peak_to_peak_v=pp,
        frequency_hz=_f(freq),
        period_s=_f(period),
        duty_cycle=_f(duty),
        rise_time_s=_f(rise),
        fall_time_s=_f(fall),
    )


def measure_digital(trace: DigitalTrace) -> DigitalMeasurements:
    tt = np.asarray(trace.transition_times, dtype=np.float64)
    init = bool(trace.initial_state)
    n = int(tt.size)
    duration = trace.end_time - trace.begin_time
    # Edge k (0-based) goes to state init ^ (k+1 odd).
    rising_mask = np.array([(not init) if k % 2 == 0 else init for k in range(n)], dtype=bool)
    rising = tt[rising_mask]
    final = init ^ bool(n % 2)

    freq = period = duty = None
    if rising.size >= 2:
        period = float((rising[-1] - rising[0]) / (rising.size - 1))
        freq = 1.0 / period

    # Time spent high, over the whole window.
    bounds = np.concatenate(([trace.begin_time], tt, [trace.end_time]))
    widths = np.diff(bounds)
    states = np.array([init ^ bool(k % 2) for k in range(widths.size)])
    if duration > 0:
        duty = float(widths[states].sum() / duration)
    inner = widths[1:-1] if widths.size > 2 else np.empty(0)

    return DigitalMeasurements(
        duration_s=float(duration),
        initial_state=int(init),
        final_state=int(final),
        transitions=n,
        rising_edges=int(rising.size),
        falling_edges=int(n - rising.size),
        frequency_hz=_f(freq),
        period_s=_f(period),
        duty_cycle=_f(duty) if n else None,
        min_pulse_s=_f(inner.min()) if inner.size else None,
        max_pulse_s=_f(inner.max()) if inner.size else None,
    )


def decimate_minmax(trace: AnalogTrace, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace to at most ``max_points`` while keeping peaks.

    Each bucket contributes its min and its max, in time order, so spikes and
    glitches survive the reduction (unlike plain striding).
    """
    v = np.asarray(trace.voltages)
    n = v.size
    t0, dt = trace.begin_time, 1.0 / trace.sample_rate
    if n <= max_points:
        return t0 + np.arange(n) * dt, v.astype(np.float64)

    buckets = max(1, max_points // 2)
    size = n // buckets
    usable = size * buckets
    blocks = v[:usable].reshape(buckets, size)
    imin = blocks.argmin(axis=1)
    imax = blocks.argmax(axis=1)
    base = np.arange(buckets) * size
    first = np.minimum(imin, imax) + base
    second = np.maximum(imin, imax) + base
    idx = np.empty(buckets * 2, dtype=np.int64)
    idx[0::2], idx[1::2] = first, second
    return t0 + idx * dt, v[idx].astype(np.float64)
