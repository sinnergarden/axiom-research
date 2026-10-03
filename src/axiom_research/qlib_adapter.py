"""Research's thin consumer of immutable Data-owned Qlib exports (P04).

Qlib uses process-global providers. Activate explicitly once per task; native
reads reject a provider switch rather than silently reading another data root.
Feature expressions and training remain Research responsibilities.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


class QlibView:
    """Verify a Data export once, then read native fields with the actual Qlib API.

    Loading does not initialize Qlib. activate() changes its process-global
    provider and clears its caches; callers own task isolation. No supplier
    calls, PIT reinterpretation, unit conversion, filling or feature computation.
    Install axiom-data[qlib] to enable this optional consumer runtime.
    """

    def __init__(self, path):
        from axiom_data import verify_qlib_export
        self.path = Path(path).resolve()
        self.manifest = verify_qlib_export(self.path)

    @property
    def reference(self):
        """Recoverable P04 identity and originating Snapshot/query, without paths."""
        return deepcopy({k:self.manifest[k] for k in (
            "schema_version", "view_id", "snapshot_id", "queries", "fields",
            "instrument_map", "calendar", "universe_query", "universe_name",
            "numeric_format", "reader_version", "exporter_version", "limitations")})

    def activate(self):
        """Select this local immutable provider, single worker and memory caches.

        Qlib initialization is explicit; its disk expression/dataset caches are
        disabled so activation/reads do not modify the immutable export directory.
        """
        import qlib
        qlib.init(provider_uri=str(self.path), region="cn", kernels=1,
                  expression_cache=None, dataset_cache=None)
        return self

    def read(self, *, fields, symbols=None, start=None, end=None, universe=None):
        """Return Qlib's keyed DataFrame for exported native fields only.

        fields uses manifest aliases without '$'. symbols are stable Data IDs;
        a named universe instead applies the exported inclusive membership
        intervals. Reads stay inside the view calendar (which includes lookback).
        Keep reference alongside the result; per-key provenance is available
        from the original Data Reader query. This is P04, not a reconstructed
        DataBatch with invented per-key PIT metadata.
        """
        from qlib.config import C
        from qlib.data import D
        configured = C.get("provider_uri", {})
        paths = configured.values() if isinstance(configured, dict) else (configured,)
        if str(self.path) not in {str(Path(p).expanduser().resolve()) for p in paths}:
            raise ValueError("activate this QlibView before reading; another task switched the provider")
        fields = tuple(fields); mapping = self.manifest["instrument_map"]
        if not fields or len(set(fields)) != len(fields) or set(fields)-set(self.manifest["fields"]):
            raise ValueError("Qlib native read requires exported field aliases")
        calendar = self.manifest["calendar"]
        start = start or calendar[0]; end = end or calendar[-1]
        if start not in calendar or end not in calendar or start > end:
            raise ValueError("Qlib read range must stay in its exported open calendar")
        if universe is not None:
            if symbols is not None or universe != self.manifest["universe_name"]:
                raise ValueError("choose exported named universe or explicit symbols")
            instruments = D.instruments(market=universe)
        else:
            symbols = tuple(mapping) if symbols is None else tuple(symbols)
            if not symbols or len(set(symbols)) != len(symbols) or set(symbols)-set(mapping):
                raise ValueError("requested stable identity is outside this Qlib export")
            instruments = [mapping[s] for s in symbols]
        return D.features(instruments=instruments, fields=["$"+f for f in fields],
                          start_time=start, end_time=end, freq="day", disk_cache=0)
