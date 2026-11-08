"""Application service: the single entry point the MCP layer talks to.

Serialises device access, persists captures, and turns raw data into compact
JSON-ready results.
"""

from __future__ import annotations

import csv
import math
import threading
from dataclasses import asdict
from typing import Any

import numpy as np

from saleae_mso_mcp import analysis
from saleae_mso_mcp.backends import Backend
from saleae_mso_mcp.domain import (
    AnalogChannelData,
    CaptureData,
    CaptureRequest,
    DigitalChannelData,
)
from saleae_mso_mcp.errors import InvalidRequestError, MsoMcpError, NotFoundError
from saleae_mso_mcp.settings import Settings
from saleae_mso_mcp.store import CaptureRecord, CaptureStore

SIG_DIGITS = 6


def _r(x: Any) -> Any:
    """Round floats to a fixed number of significant digits, recursively."""
    if isinstance(x, float):
        return float(f"{x:.{SIG_DIGITS}g}") if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_r(v) for v in x]
    return x


class MsoService:
    def __init__(self, settings: Settings, backend: Backend, store: CaptureStore | None = None):
        self.settings = settings
        self.backend = backend
        self.store = store or CaptureStore(settings.capture_dir)
        self._device_lock = threading.Lock()
        self._data_lock = threading.Lock()

    # -- device ------------------------------------------------------------

    def device_info(self) -> dict:
        with self._device_lock:
            info = self.backend.info()
        return {**asdict(info), "capture_dir": str(self.store.root)}

    def release_device(self) -> dict:
        with self._device_lock:
            self.backend.close()
        return {"released": True}

    def capture(self, request: CaptureRequest) -> dict:
        request = self._resolve_rate(request)
        record = self.store.new_record(self.backend.name, request)
        try:
            with self._device_lock:
                data = self.backend.capture(request, record.path, self.settings.capture_timeout_s)
        except BaseException:
            self.store.discard(record)
            raise
        self.store.commit(record, data)
        return self._summary(record, data)

    def _resolve_rate(self, request: CaptureRequest) -> CaptureRequest:
        """Fill in an omitted sample rate: the device maximum, divided down to fit the sample budget."""
        if request.sample_rate_hz is not None:
            return request
        top = self.settings.max_sample_rate_hz
        div = max(1, math.ceil(top * request.window_s / self.settings.auto_max_samples))
        return request.model_copy(update={"sample_rate_hz": top / div})

    # -- captures ----------------------------------------------------------

    def list_captures(self, limit: int = 20) -> dict:
        items = [
            {
                "id": r.id,
                "created": r.created.isoformat(timespec="seconds"),
                "label": r.label,
                "backend": r.backend,
                "mode": r.request.mode.kind,
                "channels": [c.label for c in r.request.analog] + [c.label for c in r.request.digital],
            }
            for r in self.store.list(limit)
        ]
        return {"captures": items}

    def summary(self, capture_id: str) -> dict:
        record, data = self._load(capture_id)
        return self._summary(record, data)

    def delete(self, capture_id: str) -> dict:
        with self._data_lock:
            self.store.delete(capture_id)
        return {"deleted": capture_id}

    # -- analysis ----------------------------------------------------------

    def measure(self, capture_id: str, channel: str, start_s: float | None, stop_s: float | None) -> dict:
        _, data = self._load(capture_id)
        ch = self._channel(data, channel)
        if isinstance(ch, AnalogChannelData):
            segs = [t.window(start_s, stop_s) for t in ch.traces]
            res = [analysis.measure_analog(s).as_dict() | {"start_s": s.begin_time} for s in segs if len(s.voltages)]
        else:
            segs = [t.window(start_s, stop_s) for t in ch.traces]
            res = [
                analysis.measure_digital(s).as_dict() | {"start_s": s.begin_time}
                for s in segs
                if s.end_time > s.begin_time
            ]
        if not res:
            raise InvalidRequestError("the requested time window contains no samples")
        out: dict[str, Any] = {"channel": channel, "type": "analog" if isinstance(ch, AnalogChannelData) else "digital"}
        if len(res) == 1:
            out |= res[0]
        else:
            out["segments"] = res
        return _r(out)

    def waveform(
        self, capture_id: str, channel: str, start_s: float | None, stop_s: float | None, max_points: int | None
    ) -> dict:
        _, data = self._load(capture_id)
        ch = self._channel(data, channel)
        if not isinstance(ch, AnalogChannelData):
            raise InvalidRequestError(f"{channel!r} is digital; use get_digital_transitions")
        limit = min(max_points or self.settings.max_waveform_points, self.settings.max_waveform_points)
        segs = [t.window(start_s, stop_s) for t in ch.traces]
        segs = [s for s in segs if len(s.voltages)]
        if not segs:
            raise InvalidRequestError("the requested time window contains no samples")
        per = max(4, limit // len(segs))
        out_segs = []
        for s in segs:
            t, v = analysis.decimate_minmax(s, per)
            out_segs.append(
                {
                    "start_s": s.begin_time,
                    "sample_rate_hz": s.sample_rate,
                    "source_samples": len(s.voltages),
                    "decimated": len(s.voltages) > per,
                    "points": [[_r(float(a)), _r(float(b))] for a, b in zip(t, v, strict=True)],
                }
            )
        return _r({"channel": channel, "units": ["s", "V"], "segments": out_segs})

    def transitions(
        self, capture_id: str, channel: str, start_s: float | None, stop_s: float | None, max_transitions: int | None
    ) -> dict:
        _, data = self._load(capture_id)
        ch = self._channel(data, channel)
        if not isinstance(ch, DigitalChannelData):
            raise InvalidRequestError(f"{channel!r} is analog; use get_waveform")
        limit = min(max_transitions or self.settings.max_transitions, self.settings.max_transitions)
        out_segs = []
        for tr in ch.traces:
            w = tr.window(start_s, stop_s)
            tt = w.transition_times
            out_segs.append(
                {
                    "start_s": w.begin_time,
                    "stop_s": w.end_time,
                    "initial_state": int(w.initial_state),
                    "total_transitions": int(tt.size),
                    "truncated": bool(tt.size > limit),
                    "times_s": [_r(float(x)) for x in tt[:limit]],
                }
            )
        return _r({"channel": channel, "segments": out_segs})

    def export_csv(self, capture_id: str, channel: str, start_s: float | None, stop_s: float | None) -> dict:
        record, data = self._load(capture_id)
        ch = self._channel(data, channel)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in channel)
        path = record.path / f"{safe}.csv"
        rows = 0
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            if isinstance(ch, AnalogChannelData):
                w.writerow(["time_s", "volts"])
                for tr in ch.traces:
                    s = tr.window(start_s, stop_s)
                    t = s.begin_time + np.arange(len(s.voltages)) / s.sample_rate
                    w.writerows(zip(t.tolist(), s.voltages.tolist(), strict=True))
                    rows += len(s.voltages)
            else:
                w.writerow(["time_s", "state"])
                for tr in ch.traces:
                    s = tr.window(start_s, stop_s)
                    w.writerow([s.begin_time, int(s.initial_state)])
                    state = s.initial_state
                    for t in s.transition_times.tolist():
                        state = not state
                        w.writerow([t, int(state)])
                    rows += 1 + s.transition_times.size
        return {"path": str(path), "rows": int(rows)}

    # -- helpers -----------------------------------------------------------

    def _load(self, capture_id: str) -> tuple[CaptureRecord, CaptureData]:
        record = self.store.get(capture_id)
        with self._data_lock:
            data = self.store.cached(capture_id)
            if data is None:
                if record.backend != self.backend.name:
                    raise MsoMcpError(
                        f"capture {capture_id} was made by the {record.backend!r} backend; "
                        f"this server runs {self.backend.name!r}"
                    )
                data = self.backend.load(record.path, record.request)
                self.store.remember(capture_id, data)
        return record, data

    @staticmethod
    def _channel(data: CaptureData, name: str) -> AnalogChannelData | DigitalChannelData:
        if name in data.analog:
            return data.analog[name]
        if name in data.digital:
            return data.digital[name]
        raise NotFoundError(f"no channel {name!r}; available: {data.channel_names()}")

    def _summary(self, record: CaptureRecord, data: CaptureData) -> dict:
        channels: list[dict] = []
        for ch in data.analog.values():
            first = next((t for t in ch.traces if len(t.voltages)), None)
            entry: dict[str, Any] = {
                "name": ch.name,
                "type": "analog",
                "channel": ch.channel,
                "segments": len(ch.traces),
            }
            if first is not None:
                entry |= {
                    "start_s": first.begin_time,
                    "stop_s": ch.traces[-1].end_time,
                    "sample_rate_hz": first.sample_rate,
                    "samples": sum(len(t.voltages) for t in ch.traces),
                    "measurements": analysis.measure_analog(first).as_dict(),
                }
            channels.append(entry)
        for dch in data.digital.values():
            first_d = dch.traces[0] if dch.traces else None
            entry = {
                "name": dch.name,
                "type": "digital",
                "port": dch.port,
                "channel": dch.channel,
                "segments": len(dch.traces),
            }
            if first_d is not None:
                entry |= {
                    "start_s": first_d.begin_time,
                    "stop_s": dch.traces[-1].end_time,
                    "measurements": analysis.measure_digital(first_d).as_dict(),
                }
            channels.append(entry)
        req = record.request
        actual = next((c["sample_rate_hz"] for c in channels if "sample_rate_hz" in c), None)
        extra: dict[str, Any] = {}
        if actual is not None:
            extra["sample_rate_hz"] = actual
            if req.sample_rate_hz and abs(actual - req.sample_rate_hz) / req.sample_rate_hz > 0.01:
                extra["sample_rate_note"] = (
                    f"device used {actual / 1e6:g} MS/s instead of the requested "
                    f"{req.sample_rate_hz / 1e6:g} MS/s (it only supports certain rates)"
                )
        return _r(
            {
                "id": record.id,
                "created": record.created.isoformat(timespec="seconds"),
                "label": req.label,
                "backend": record.backend,
                "mode": req.mode.model_dump(mode="json"),
                "requested_sample_rate_hz": req.sample_rate_hz,
                **extra,
                "path": str(record.path),
                "channels": channels,
            }
        )
