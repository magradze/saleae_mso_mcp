"""Backend-neutral domain model.

Requests are pydantic models so they double as MCP tool input schemas.
Capture data are plain dataclasses holding numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class Coupling(StrEnum):
    DC = "dc"
    AC = "ac"


class ProbeAttenuation(StrEnum):
    X1 = "1x"
    X10 = "10x"


class BandwidthLimit(StrEnum):
    MHZ_20 = "20mhz"
    MHZ_350 = "350mhz"


class DownsampleMode(StrEnum):
    AVERAGE = "average"
    DROP = "drop"


class EdgeDirection(StrEnum):
    RISING = "rising"
    FALLING = "falling"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class AnalogChannelSpec(_Model):
    """One scope (analog) channel to enable."""

    channel: Annotated[int, Field(ge=0, description="Scope channel index, 0-based (CH1 = 0).")]
    name: str | None = Field(None, description="Label for the channel. Defaults to CH<n+1>.")
    center_voltage: float = Field(0.0, description="Vertical center of the input range, volts.")
    voltage_range: Annotated[float, Field(gt=0)] = Field(
        10.0, description="Full-scale input range, volts (peak to peak)."
    )
    probe: ProbeAttenuation = Field(ProbeAttenuation.X10, description="Probe attenuation setting.")
    coupling: Coupling = Coupling.DC
    bandwidth: BandwidthLimit = BandwidthLimit.MHZ_350

    @property
    def label(self) -> str:
        return self.name or f"CH{self.channel + 1}"


class DigitalChannelSpec(_Model):
    """One digital (logic probe) channel to enable."""

    port: Annotated[int, Field(ge=0, description="Digital port index.")]
    channel: Annotated[int, Field(ge=0, description="Bit within the port, 0-based.")]
    name: str | None = Field(None, description="Label. Defaults to D<port>.<channel>.")
    threshold_volts: float | None = Field(
        None, description="Logic threshold. All channels on one port share a threshold."
    )

    @property
    def label(self) -> str:
        return self.name or f"D{self.port}.{self.channel}"


class EdgeTriggerSpec(_Model):
    channel: str = Field(..., description="Name of an enabled analog channel to trigger on.")
    threshold_volts: float = 0.0
    direction: EdgeDirection = EdgeDirection.RISING
    holdoff_s: Annotated[float, Field(ge=0)] = 0.0


class TimedMode(_Model):
    kind: Literal["timed"] = "timed"
    duration_s: Annotated[float, Field(gt=0, description="Capture length, seconds.")]


class TriggeredMode(_Model):
    kind: Literal["triggered"] = "triggered"
    trigger: EdgeTriggerSpec
    pre_trigger_s: Annotated[float, Field(ge=0)] = 0.0005
    post_trigger_s: Annotated[float, Field(gt=0)] = 0.0005
    timeout_s: Annotated[float, Field(gt=0, description="Give up if no trigger.")] = 1.0


CaptureMode = Annotated[TimedMode | TriggeredMode, Field(discriminator="kind")]


class CaptureRequest(_Model):
    analog: list[AnalogChannelSpec] = Field(default_factory=list)
    digital: list[DigitalChannelSpec] = Field(default_factory=list)
    sample_rate_hz: Annotated[float | None, Field(gt=0)] = None
    downsample: DownsampleMode = DownsampleMode.AVERAGE
    mode: CaptureMode
    label: str | None = Field(None, description="Free-text note stored with the capture.")

    @model_validator(mode="after")
    def _check(self) -> CaptureRequest:
        if not self.analog and not self.digital:
            raise ValueError("enable at least one analog or digital channel")
        names = [c.label for c in self.analog] + [c.label for c in self.digital]
        if len(names) != len(set(names)):
            raise ValueError(f"channel names must be unique: {names}")
        if isinstance(self.mode, TriggeredMode) and self.mode.trigger.channel not in {c.label for c in self.analog}:
            raise ValueError("trigger channel must be one of the enabled analog channels")
        return self

    @property
    def window_s(self) -> float:
        m = self.mode
        return m.duration_s if isinstance(m, TimedMode) else m.pre_trigger_s + m.post_trigger_s


# ---------------------------------------------------------------------------
# Capture data
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnalogTrace:
    """A contiguous run of evenly spaced samples."""

    begin_time: float
    sample_rate: float
    voltages: np.ndarray

    @property
    def end_time(self) -> float:
        return self.begin_time + len(self.voltages) / self.sample_rate

    def window(self, start: float | None, stop: float | None) -> AnalogTrace:
        n = len(self.voltages)
        i0 = 0 if start is None else int(np.clip(np.ceil((start - self.begin_time) * self.sample_rate), 0, n))
        i1 = n if stop is None else int(np.clip(np.floor((stop - self.begin_time) * self.sample_rate) + 1, 0, n))
        i1 = max(i0, i1)
        return AnalogTrace(self.begin_time + i0 / self.sample_rate, self.sample_rate, self.voltages[i0:i1])


@dataclass(slots=True)
class DigitalTrace:
    begin_time: float
    end_time: float
    initial_state: bool
    transition_times: np.ndarray

    def state_at(self, t: float) -> bool:
        k = int(np.searchsorted(self.transition_times, t, side="right"))
        return bool(self.initial_state) ^ bool(k % 2)

    def window(self, start: float | None, stop: float | None) -> DigitalTrace:
        b = self.begin_time if start is None else max(start, self.begin_time)
        e = self.end_time if stop is None else min(stop, self.end_time)
        e = max(b, e)
        tt = self.transition_times
        sel = tt[(tt > b) & (tt <= e)]
        return DigitalTrace(b, e, self.state_at(b), sel)


@dataclass(slots=True)
class AnalogChannelData:
    name: str
    channel: int
    traces: list[AnalogTrace]


@dataclass(slots=True)
class DigitalChannelData:
    name: str
    port: int
    channel: int
    traces: list[DigitalTrace]


@dataclass(slots=True)
class CaptureData:
    analog: dict[str, AnalogChannelData] = field(default_factory=dict)
    digital: dict[str, DigitalChannelData] = field(default_factory=dict)

    def channel_names(self) -> list[str]:
        return [*self.analog, *self.digital]
