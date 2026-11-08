# saleae-mso-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server for the
**Saleae Logic MSO** mixed-signal oscilloscope. It lets Claude (or any MCP
client) configure the scope, run timed or edge-triggered captures, and read
back measurements and waveforms, all from a chat.

Saleae's built-in Logic 2 MCP server supports their logic analyzers but not
the Logic MSO yet. This project fills that gap by talking to the hardware
directly through Saleae's official
[`saleae-mso-api`](https://docs.saleae.com/msoapi/guides/getting_started),
with no Logic 2 GUI involved.

> Status: alpha. Developed against the Logic MSO on Linux. Feedback and PRs welcome.

## What the model gets

Raw scope data is millions of samples, far too much for a context window. The
server therefore returns *facts* first and *points* on request:

| Tool | Purpose |
|---|---|
| `get_device_info` | Is an MSO connected? Serial, part number, capture folder. |
| `capture_timed` | Capture for N seconds. |
| `capture_triggered` | Wait for an analog edge, capture around it (t = 0 at trigger). |
| `list_captures` / `get_capture_summary` | Browse saved captures. |
| `measure` | Min/max/mean/RMS/AC-RMS, Vpp, frequency, period, duty cycle, 10–90 % rise/fall, or edge stats for logic channels, over any time window. |
| `get_waveform` | `[time, volts]` points, min/max-decimated so glitches survive. |
| `get_digital_transitions` | Initial state + edge times of a logic channel. |
| `export_csv` | Full-resolution CSV written next to the capture. |
| `delete_capture` | Remove a capture. |
| `release_device` | Hand the USB device back to Logic 2. |

Every capture is saved to disk (default `~/.local/share/saleae-mso-mcp/captures`)
and can be re-analysed later, even after a restart.

## Install

Requirements: Python ≥ 3.11, a Logic MSO, and Logic 2 ≥ 2.4.20 installed once
(it provides the USB drivers / udev rules).

```bash
git clone https://github.com/giorgimagradze/saleae-mso-mcp.git
cd saleae-mso-mcp
uv sync --extra hardware
```

With plain pip:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[hardware]' --extra-index-url https://downloads.saleae.com/pypi/
```

On Linux without Logic 2, install the udev rules:
`python -m saleae.mso_api.utils.install_udev`.

## Connect to Claude

**Claude Desktop**: edit `claude_desktop_config.json`
(Linux `~/.config/Claude/`, macOS `~/Library/Application Support/Claude/`,
Windows `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "saleae-mso": {
      "command": "uv",
      "args": ["--directory", "/path/to/saleae-mso-mcp", "run", "saleae-mso-mcp"]
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add saleae-mso -- uv --directory /path/to/saleae-mso-mcp run saleae-mso-mcp
```

Restart the client, then ask something like *"capture CH1 for 10 ms at
1 GS/s and tell me the frequency and rise time"*.

> Only one program can own the MSO at a time. Close Logic 2 before capturing,
> or ask the model to call `release_device` before reopening Logic 2.

## Try it without hardware

```bash
uv run saleae-mso-mcp --backend simulated
```

The simulated backend produces deterministic sine waves on analog channels and
square waves on logic channels. It is what the test suite and CI use.

## Configuration

| Env var | CLI flag | Default |
|---|---|---|
| `SALEAE_MSO_BACKEND` | `--backend` | `hardware` (`simulated` for no device) |
| `SALEAE_MSO_SERIAL` | `--serial` | first MSO found |
| `SALEAE_MSO_CAPTURE_DIR` | `--capture-dir` | `~/.local/share/saleae-mso-mcp/captures` |
| `SALEAE_MSO_CAPTURE_TIMEOUT` | | `60` seconds |
| `SALEAE_MSO_MAX_POINTS` | | `2000` waveform points per call |
| `SALEAE_MSO_MAX_TRANSITIONS` | | `1000` edges per call |
| `SALEAE_MSO_MAX_SAMPLE_RATE` | | `1.6e9`; used when a capture omits `sample_rate_hz` |
| `SALEAE_MSO_AUTO_MAX_SAMPLES` | | `20000000` samples per channel before that rate is divided down |

`--transport streamable-http` runs the server over HTTP instead of stdio.

## Architecture

```
saleae_mso_mcp/
├── server.py        MCP tools: thin layer, validation + error mapping
├── service.py       application service: locking, persistence, result shaping
├── analysis.py      pure numpy measurements and min/max decimation
├── domain.py        backend-neutral requests (pydantic) and capture data
├── store.py         on-disk capture repository + small LRU cache
├── settings.py      env/CLI configuration
├── cli.py           entry point
└── backends/
    ├── base.py      Backend protocol
    ├── hardware.py  Logic MSO via saleae-mso-api (imported lazily)
    └── simulated.py synthetic signals for tests and demos
```

Adding support for another instrument means writing one more backend.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT © Giorgi Magradze. Not affiliated with Saleae, Inc. "Saleae" and "Logic MSO"
are trademarks of their respective owners.
