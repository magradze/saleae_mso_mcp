"""MCP surface: tool definitions that delegate to ``MsoService``."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from saleae_mso_mcp import __version__
from saleae_mso_mcp.domain import (
    AnalogChannelSpec,
    CaptureRequest,
    DigitalChannelSpec,
    DownsampleMode,
    EdgeTriggerSpec,
    TimedMode,
    TriggeredMode,
)
from saleae_mso_mcp.errors import MsoMcpError
from saleae_mso_mcp.service import MsoService

INSTRUCTIONS = """\
Controls a Saleae Logic MSO (mixed-signal oscilloscope).

Workflow: get_device_info -> capture_timed or capture_triggered -> inspect the
returned summary (it already contains min/max/mean/RMS, frequency, duty cycle,
rise/fall time per channel) -> drill in with measure, get_waveform (min/max
decimated points) or get_digital_transitions using start_s/stop_s windows.

Times are in seconds. For triggered captures t = 0 is the trigger point.
Analog channel indices are 0-based (CH1 = 0). Only one program can own the
device: if Logic 2 is open, captures fail; call release_device before handing
the device back to Logic 2.
"""

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

ChannelName = Annotated[str, Field(description="Channel name as shown in the capture summary.")]
CaptureId = Annotated[str, Field(description="Capture id from a capture call or list_captures.")]
StartS = Annotated[float | None, Field(description="Window start, seconds. Omit for capture start.")]
StopS = Annotated[float | None, Field(description="Window end, seconds. Omit for capture end.")]
SampleRate = Annotated[
    float | None,
    Field(
        gt=0,
        description="Analog sample rate. Omit for the highest rate (1.6 GS/s), divided down only for long "
        "captures. The device may round to a supported rate; the summary reports the rate used.",
    ),
]


def _tool_errors(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except MsoMcpError as exc:
            raise ToolError(str(exc)) from exc
        except ValueError as exc:
            raise ToolError(f"invalid request: {exc}") from exc
        except Exception as exc:
            log.exception("unexpected error in %s", fn.__name__)
            raise ToolError(f"internal error: {type(exc).__name__}: {exc}") from exc

    return wrapper  # type: ignore[return-value]


def build_server(service: MsoService) -> MCPServer:
    mcp = MCPServer(name="saleae-mso", instructions=INSTRUCTIONS, version=__version__)

    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    device = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    @mcp.tool(annotations=read_only)
    @_tool_errors
    def get_device_info() -> dict:
        """Report whether a Logic MSO is connected, its serial and part number, and where captures are saved."""
        return service.device_info()

    @mcp.tool(annotations=device)
    @_tool_errors
    def capture_timed(
        duration_s: Annotated[float, Field(gt=0, description="Capture length in seconds.")],
        analog: Annotated[list[AnalogChannelSpec], Field(description="Scope channels to enable.")] = [],  # noqa: B006
        digital: Annotated[list[DigitalChannelSpec], Field(description="Logic channels to enable.")] = [],  # noqa: B006
        sample_rate_hz: SampleRate = None,
        downsample: DownsampleMode = DownsampleMode.AVERAGE,
        label: str | None = None,
    ) -> dict:
        """Capture for a fixed duration and return a per-channel measurement summary.

        If you set sample_rate_hz, keep duration_s * sample_rate_hz to tens of millions of samples at most.
        """
        req = CaptureRequest(
            analog=analog,
            digital=digital,
            sample_rate_hz=sample_rate_hz,
            downsample=downsample,
            mode=TimedMode(duration_s=duration_s),
            label=label,
        )
        return service.capture(req)

    @mcp.tool(annotations=device)
    @_tool_errors
    def capture_triggered(
        trigger: EdgeTriggerSpec,
        analog: Annotated[
            list[AnalogChannelSpec],
            Field(min_length=1, description="Scope channels; must include the trigger channel."),
        ],
        digital: Annotated[list[DigitalChannelSpec], Field(description="Logic channels to enable.")] = [],  # noqa: B006
        pre_trigger_s: Annotated[float, Field(ge=0)] = 0.0005,
        post_trigger_s: Annotated[float, Field(gt=0)] = 0.0005,
        timeout_s: Annotated[float, Field(gt=0, description="Give up if no trigger occurs.")] = 1.0,
        sample_rate_hz: SampleRate = None,
        downsample: DownsampleMode = DownsampleMode.AVERAGE,
        label: str | None = None,
    ) -> dict:
        """Wait for an analog edge trigger, capture around it (t = 0 at the trigger) and summarise."""
        req = CaptureRequest(
            analog=analog,
            digital=digital,
            sample_rate_hz=sample_rate_hz,
            downsample=downsample,
            mode=TriggeredMode(
                trigger=trigger, pre_trigger_s=pre_trigger_s, post_trigger_s=post_trigger_s, timeout_s=timeout_s
            ),
            label=label,
        )
        return service.capture(req)

    @mcp.tool(annotations=read_only)
    @_tool_errors
    def list_captures(limit: Annotated[int, Field(ge=1, le=200)] = 20) -> dict:
        """List saved captures, newest first."""
        return service.list_captures(limit)

    @mcp.tool(annotations=read_only)
    @_tool_errors
    def get_capture_summary(capture_id: CaptureId) -> dict:
        """Settings and per-channel measurements of a saved capture."""
        return service.summary(capture_id)

    @mcp.tool(annotations=read_only)
    @_tool_errors
    def measure(capture_id: CaptureId, channel: ChannelName, start_s: StartS = None, stop_s: StopS = None) -> dict:
        """Measure one channel over a time window.

        Analog: min/max/mean/RMS/AC-RMS, peak-to-peak, frequency, period, duty cycle, rise/fall (10-90%).
        Digital: edge counts, frequency, duty cycle, min/max pulse width.
        """
        return service.measure(capture_id, channel, start_s, stop_s)

    @mcp.tool(annotations=read_only)
    @_tool_errors
    def get_waveform(
        capture_id: CaptureId,
        channel: ChannelName,
        start_s: StartS = None,
        stop_s: StopS = None,
        max_points: Annotated[int | None, Field(ge=4, description="Upper bound on returned points.")] = 500,
    ) -> dict:
        """Return [time_s, volts] points for an analog channel.

        Long windows are reduced with min/max decimation so spikes and glitches are kept.
        Narrow the window to see full resolution.
        """
        return service.waveform(capture_id, channel, start_s, stop_s, max_points)

    @mcp.tool(annotations=read_only)
    @_tool_errors
    def get_digital_transitions(
        capture_id: CaptureId,
        channel: ChannelName,
        start_s: StartS = None,
        stop_s: StopS = None,
        max_transitions: Annotated[int | None, Field(ge=1)] = 200,
    ) -> dict:
        """Initial state and transition times of a digital channel within a window."""
        return service.transitions(capture_id, channel, start_s, stop_s, max_transitions)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
    @_tool_errors
    def export_csv(capture_id: CaptureId, channel: ChannelName, start_s: StartS = None, stop_s: StopS = None) -> dict:
        """Write full-resolution channel data to a CSV file inside the capture folder and return its path."""
        return service.export_csv(capture_id, channel, start_s, stop_s)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True))
    @_tool_errors
    def delete_capture(capture_id: CaptureId) -> dict:
        """Permanently delete a saved capture and its files."""
        return service.delete(capture_id)

    @mcp.tool(annotations=device)
    @_tool_errors
    def release_device() -> dict:
        """Close the USB connection so Logic 2 or another program can use the MSO. Reopens on the next capture."""
        return service.release_device()

    return mcp
