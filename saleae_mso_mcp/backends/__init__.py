"""Device backends."""

from __future__ import annotations

from saleae_mso_mcp.backends.base import Backend, DeviceInfo
from saleae_mso_mcp.settings import BackendKind, Settings

__all__ = ["Backend", "DeviceInfo", "create_backend"]


def create_backend(settings: Settings) -> Backend:
    if settings.backend is BackendKind.SIMULATED:
        from saleae_mso_mcp.backends.simulated import SimulatedBackend

        return SimulatedBackend()
    from saleae_mso_mcp.backends.hardware import HardwareBackend

    return HardwareBackend(settings.serial_number)
