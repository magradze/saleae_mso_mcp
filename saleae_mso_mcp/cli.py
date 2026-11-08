"""Command-line entry point: ``saleae-mso-mcp``."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from saleae_mso_mcp import __version__
from saleae_mso_mcp.backends import create_backend
from saleae_mso_mcp.server import build_server
from saleae_mso_mcp.service import MsoService
from saleae_mso_mcp.settings import BackendKind, Settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="saleae-mso-mcp", description="MCP server for Saleae Logic MSO.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--backend", choices=[b.value for b in BackendKind], help="hardware (default) or simulated")
    p.add_argument("--serial", help="serial number of the MSO to use (default: first found)")
    p.add_argument("--capture-dir", type=Path, help="where captures are stored")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # stdout carries the MCP protocol on stdio; logs go to stderr.
    logging.basicConfig(
        level=args.log_level.upper(), stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = Settings.from_env()
    if args.backend:
        settings = replace(settings, backend=BackendKind(args.backend))
    if args.serial:
        settings = replace(settings, serial_number=args.serial)
    if args.capture_dir:
        settings = replace(settings, capture_dir=args.capture_dir.expanduser())

    service = MsoService(settings, create_backend(settings))
    try:
        build_server(service).run(transport=args.transport)
    finally:
        service.release_device()


if __name__ == "__main__":
    main()
