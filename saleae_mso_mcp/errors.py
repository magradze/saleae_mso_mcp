"""Error types. Anything derived from ``MsoMcpError`` reaches the model as a readable message."""

from __future__ import annotations


class MsoMcpError(Exception):
    """Expected failure with a message meant for the user or model."""


class DeviceUnavailableError(MsoMcpError):
    pass


class CaptureFailedError(MsoMcpError):
    pass


class NotFoundError(MsoMcpError):
    pass


class InvalidRequestError(MsoMcpError):
    pass
