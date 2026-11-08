"""End-to-end through the MCP layer, using the SDK's in-memory client."""

from __future__ import annotations

import json

import anyio
import pytest

from saleae_mso_mcp.server import build_server

EXPECTED_TOOLS = {
    "get_device_info",
    "capture_timed",
    "capture_triggered",
    "list_captures",
    "get_capture_summary",
    "measure",
    "get_waveform",
    "get_digital_transitions",
    "export_csv",
    "delete_capture",
    "release_device",
}


def _payload(result) -> dict:
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


def run(coro_fn):
    return anyio.run(coro_fn)


def test_tools_listed(service):
    mcp = build_server(service)
    tools = run(mcp.list_tools)
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_capture_and_measure_via_mcp(service):
    mcp = build_server(service)

    async def flow():
        r = await mcp.call_tool(
            "capture_timed",
            {
                "duration_s": 0.005,
                "sample_rate_hz": 1e6,
                "analog": [{"channel": 2, "name": "sig"}],
            },
        )
        cid = _payload(r)["id"]
        m = await mcp.call_tool("measure", {"capture_id": cid, "channel": "sig"})
        return _payload(m)

    assert run(flow)["frequency_hz"] == pytest.approx(3000, rel=1e-3)


def test_expected_errors_are_readable(service):
    mcp = build_server(service)

    async def flow():
        return await mcp.call_tool("get_capture_summary", {"capture_id": "missing"})

    try:
        r = run(flow)
    except Exception as exc:  # SDK may raise ToolError directly
        assert "no capture" in str(exc)
    else:
        assert r.is_error and "no capture" in r.content[0].text
