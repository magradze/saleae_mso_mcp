from __future__ import annotations

import pytest

from saleae_mso_mcp.backends.simulated import SimulatedBackend
from saleae_mso_mcp.service import MsoService
from saleae_mso_mcp.settings import BackendKind, Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(backend=BackendKind.SIMULATED, capture_dir=tmp_path / "captures")


@pytest.fixture
def service(settings):
    return MsoService(settings, SimulatedBackend(seed=1))
