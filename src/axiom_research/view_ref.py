"""Research-owned logical ViewRef over a concrete Data Snapshot and query.

The established Research manifests accept ArtifactRef. `as_artifact_ref`
encodes this inline definition in metadata and binds its exact JSON by digest,
so those manifests need no schema migration or persisted View file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping

from .contracts import ArtifactRef


@dataclass(frozen=True)
class ViewRef:
    snapshot_id: str
    domain: str
    query: Mapping[str, Any]
    reader_version: str
    derivation: Mapping[str, Any] | None = None
    _payload: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.snapshot_id in ("", "current", "latest") or not self.domain or not self.reader_version:
            raise ValueError("logical ViewRef requires a concrete Snapshot and Reader version")
        if not isinstance(self.query, Mapping) or not all(
                k in self.query for k in ("fields", "symbols", "pit_policy", "purpose")):
            raise ValueError("logical ViewRef requires complete query context")
        if "sessions" in self.query:
            if set(self.query["sessions"]) != set(self.query.get("cutoff_by_session", {})):
                raise ValueError("logical ViewRef cutoffs do not cover sessions")
        elif not all(k in self.query for k in ("start", "end", "cutoff", "time_field", "filters")):
            raise ValueError("logical ViewRef requires complete EventQuery context")
        payload = {"snapshot_id": self.snapshot_id, "domain": self.domain,
                   "query": dict(self.query), "reader_version": self.reader_version,
                   "derivation": dict(self.derivation) if self.derivation is not None else None}
        object.__setattr__(self, "_payload", json.dumps(payload, ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self) -> dict:
        return json.loads(self._payload)

    @property
    def digest(self) -> str:
        return "sha256:" + sha256(self._payload.encode()).hexdigest()

    @classmethod
    def from_batch(cls, batch: Any) -> "ViewRef":
        context = batch.to_json()["context"]
        if context.get("contract_version") != "data_batch_v1":
            raise ValueError("unsupported DataBatch contract")
        derivation = context.get("derivation")
        if context.get("event_reader_version"):
            derivation = {**(derivation or {}), "event_reader_version": context["event_reader_version"]}
        return cls(context["snapshot_id"], context["domain"],
                   context["query"], context["reader_version"], derivation)

    def as_artifact_ref(self) -> ArtifactRef:
        """Legacy manifest-compatible inline reference; no View bytes published."""
        return ArtifactRef(artifact_type="LogicalDataView", artifact_id=self.digest[7:],
                           artifact_contract_version="1", content_digest=self.digest,
                           uri="inline:logical-data-view", metadata={"logical_view": self.to_dict()})
