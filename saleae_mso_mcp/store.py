"""On-disk capture repository with a small in-memory cache.

Layout::

    <capture_dir>/<capture_id>/capture.json   metadata + the original request
    <capture_dir>/<capture_id>/...            backend-specific data files
"""

from __future__ import annotations

import json
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from saleae_mso_mcp.domain import CaptureData, CaptureRequest
from saleae_mso_mcp.errors import NotFoundError

META_FILE = "capture.json"
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    id: str
    path: Path
    created: datetime
    backend: str
    request: CaptureRequest

    @property
    def label(self) -> str | None:
        return self.request.label

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "created": self.created.isoformat(),
            "backend": self.backend,
            "request": self.request.model_dump(mode="json"),
        }

    @classmethod
    def from_json(cls, path: Path, d: dict) -> CaptureRecord:
        return cls(
            id=d["id"],
            path=path,
            created=datetime.fromisoformat(d["created"]),
            backend=d["backend"],
            request=CaptureRequest.model_validate(d["request"]),
        )


class CaptureStore:
    def __init__(self, root: Path, cache_size: int = 4) -> None:
        self.root = root
        self._cache: OrderedDict[str, CaptureData] = OrderedDict()
        self._cache_size = cache_size

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # -- records -----------------------------------------------------------

    def new_record(self, backend: str, request: CaptureRequest) -> CaptureRecord:
        self.ensure_root()
        now = datetime.now(UTC)
        base = now.strftime("%Y%m%d-%H%M%S")
        cid, n = base, 1
        while (self.root / cid).exists():
            n += 1
            cid = f"{base}-{n}"
        path = self.root / cid
        path.mkdir()
        return CaptureRecord(cid, path, now, backend, request)

    def commit(self, record: CaptureRecord, data: CaptureData) -> None:
        (record.path / META_FILE).write_text(json.dumps(record.to_json(), indent=2))
        self._remember(record.id, data)

    def discard(self, record: CaptureRecord) -> None:
        shutil.rmtree(record.path, ignore_errors=True)

    def get(self, capture_id: str) -> CaptureRecord:
        if not _ID_RE.match(capture_id):
            raise NotFoundError(f"invalid capture id {capture_id!r}")
        path = self.root / capture_id
        meta = path / META_FILE
        if not meta.is_file():
            raise NotFoundError(f"no capture with id {capture_id!r}; use list_captures")
        return CaptureRecord.from_json(path, json.loads(meta.read_text()))

    def list(self, limit: int = 50) -> list[CaptureRecord]:
        if not self.root.is_dir():
            return []
        out: list[CaptureRecord] = []
        for meta in sorted(self.root.glob(f"*/{META_FILE}"), reverse=True):
            try:
                out.append(CaptureRecord.from_json(meta.parent, json.loads(meta.read_text())))
            except Exception:
                continue
            if len(out) >= limit:
                break
        return out

    def delete(self, capture_id: str) -> None:
        rec = self.get(capture_id)
        self._cache.pop(capture_id, None)
        shutil.rmtree(rec.path)

    # -- data cache --------------------------------------------------------

    def cached(self, capture_id: str) -> CaptureData | None:
        data = self._cache.get(capture_id)
        if data is not None:
            self._cache.move_to_end(capture_id)
        return data

    def _remember(self, capture_id: str, data: CaptureData) -> None:
        self._cache[capture_id] = data
        self._cache.move_to_end(capture_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def remember(self, capture_id: str, data: CaptureData) -> None:
        self._remember(capture_id, data)
