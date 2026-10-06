"""Small orchestration fixture: immutable refs and original per-session cutoffs."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import digest
from axiom_research.stock_ml import (_prepare_stock_inputs, _prepare_stock_features,
                                     _prepare_stock_labels)


class StockInputReuseTests(unittest.TestCase):
    def test_identity_and_view_reference_reuse_preserves_cutoff_and_output_bindings(self):
        import pandas as pd
        symbols = ("A", "B")
        sessions = ("2024-01-02", "2024-01-03")
        fields = ("open", "high", "low", "close", "amount_cny", "factor")
        cutoffs = {session: session + "T20:30:00Z" for session in sessions}
        config = dict(snapshot="s-fixed", symbols=list(symbols), read_sessions=list(sessions),
            feature_sessions=list(sessions), calendar=list(sessions), cutoff_by_session=cutoffs,
            pit_policy="best_effort_vendor_v1", universe_id="csi300", scope_ref="fixed-scope",
            feature_selection=[{"id": "MOM010", "semantic_version": "1"}],
            fit_cutoff=cutoffs[sessions[0]], evaluation_cutoff=cutoffs[sessions[-1]],
            prediction_sessions=[sessions[-1]])
        reference = {"view_id": "fixed-view", "instrument_map": {"A": "a", "B": "b"}}
        native = pd.DataFrame([[1.0] * len(fields)] * (len(symbols) * len(sessions)),
            index=pd.MultiIndex.from_tuples([(security.lower(), pd.Timestamp(session))
                for session in sessions for security in symbols]), columns=["$" + f for f in fields])

        class View:
            reference_reads = 0

            def __init__(self, path): pass
            def activate(self): return self
            def read(self, **kwargs): return native

            @property
            def reference(self):
                View.reference_reads += 1
                return deepcopy(reference)

        class Document:
            def __init__(self, wire):
                self.wire, self.identity_reads = wire, 0

            @property
            def identity(self):
                self.identity_reads += 1
                return digest(self.wire)

            def to_dict(self): return deepcopy(self.wire)

        class Batch:
            def __init__(self, query): self.query = query
            def to_json(self):
                return {"records": [{"security_id": symbol, "session": session, "is_member": True}
                    for session in self.query.sessions for symbol in self.query.symbols]}

        class Data:
            def __init__(self): self.queries, self.member_queries = [], []
            def export_qlib(self, **kwargs): pass
            def read(self, *, snapshot, query):
                self.queries.append(query)
                return Batch(query)
            def members(self, *, snapshot, query):
                self.member_queries.append(query)
                return Batch(query)

        plans, frames = [], []

        def adapt(wire, reference_wire, **kwargs):
            plan = Document({"session": max(row['session'] for row in wire['records'])})
            plans.append(plan)
            return SimpleNamespace(plan=plan, facts=Document({"facts": True}),
                context=Document({"context": True}), source_evidence={})

        def execute(plan, facts, context):
            frame = Document({"rows": [{"security_id": symbol, "values": [float(index)],
                "availability": [cutoffs[plan.wire["session"]]], "valid": [True], "reasons": [[]]}
                for index, symbol in enumerate(symbols)]})
            frames.append(frame)
            return frame

        data = Data()
        catalog = SimpleNamespace(identity="fixed-catalog", recipe_ref=lambda *args, **kwargs: "fixed-recipe")
        def label(batch, *, calendar, feature_sessions):
            return {"rows": [], "calendar": list(calendar), "feature_sessions": list(feature_sessions),
                    "sessions": list(batch.query.sessions), "cutoffs": dict(batch.query.cutoff_by_session)}
        with patch("axiom_research.qlib_adapter.QlibView", View), \
             patch("axiom_research.stock_ml._project_qlib", side_effect=lambda batch, values, ref: batch) as project, \
             patch("axiom_research.stock_ml._adjust_feature", side_effect=lambda price, factor, session, **kw: (price,price.to_json())), \
             patch("axiom_research.data_adapter._adapt_decision_wires", side_effect=adapt), \
             patch("axiom_research.feature_catalog.build_feature_plan", side_effect=lambda plan, *args, **kwargs: plan), \
             patch("axiom_engine.core.execute_feature_plan", side_effect=execute), \
             patch("axiom_data.adjust_prices", side_effect=lambda price, factor, **kwargs: price) as adjust, \
             patch("axiom_research.labels.build_forward_labels", side_effect=label), \
             patch("axiom_research.stock_ml.time.perf_counter", return_value=0):
            features, training, evaluation, inputs, stats = _prepare_stock_inputs(data, config=config,
                destination="unused", catalog=catalog, chosen=[{"id": "MOM010"}], progress=None)
            self.assertEqual(View.reference_reads, 1)
            separate_data = Data()
            feature_only_config = {**config, "prediction_sessions": list(sessions),
                                   "fit_cutoff": "2024-01-01T20:30:00Z"}
            with patch("axiom_research.stock_ml._prepare_stock_labels", side_effect=AssertionError("outcomes")), \
                 patch("axiom_data.adjust_prices", side_effect=AssertionError("outcome adjustment")), \
                 patch("axiom_research.labels.build_forward_labels", side_effect=AssertionError("outcome labels")):
                isolated_features, isolated_inputs, isolated_stats = _prepare_stock_features(separate_data,
                    config=feature_only_config, destination="unused", catalog=catalog,
                    chosen=[{"id": "MOM010"}], progress=None)
            self.assertEqual(isolated_stats["data_read_calls"], 6)
            isolated_labels = _prepare_stock_labels(separate_data, config=config,
                training_sessions=[sessions[0]], metrics=isolated_stats)
        self.assertEqual((isolated_features, isolated_inputs, isolated_stats), (features, inputs, stats))
        self.assertEqual(isolated_labels, (training, evaluation))
        self.assertEqual(separate_data.queries, data.queries)
        self.assertEqual(separate_data.member_queries, data.member_queries)
        self.assertEqual(View.reference_reads, 2)
        self.assertEqual([plan.identity_reads for plan in plans], [1, 1, 1, 1])
        self.assertEqual([frame.identity_reads for frame in frames], [1, 1, 1, 1])
        self.assertEqual(features["qlib_view"], reference)
        self.assertEqual(stats["feature_core_calls"], 2)
        self.assertEqual(stats["data_read_calls"], 10)
        self.assertEqual(stats["core_calls"], 2)
        self.assertEqual([call.args[2] for call in project.call_args_list], ["fixed-view"] * 8)
        for session, plan, frame, item in zip(sessions, plans, frames, inputs):
            for row in [r for r in features["rows"] if r["session"] == session]:
                self.assertEqual(row["source_refs"], [digest(frame.wire), digest(plan.wire)])
            self.assertEqual(item["core_frame_ref"], digest(frame.wire))
        for index, session in enumerate(sessions):
            for query in data.queries[index * 2:index * 2 + 2]:
                self.assertEqual(query.sessions, sessions[:index + 1])
                self.assertEqual(query.cutoff_by_session,
                                 {historical: cutoffs[session] for historical in query.sessions})
        for offset, cutoff, allowed in ((4, config["fit_cutoff"], sessions[:1]),
                                        (6, config["evaluation_cutoff"], sessions)):
            prices, factors = data.queries[offset:offset + 2]
            self.assertEqual(prices.fields, ("open", "close"))
            self.assertEqual(factors.fields, ("factor",))
            self.assertEqual((prices.domain, factors.domain), ("market_daily", "adjustment_factors"))
            for query in (prices, factors):
                self.assertEqual(query.purpose, "label_outcomes")
                self.assertEqual(query.sessions, allowed)
                self.assertEqual(query.cutoff_by_session, {session: cutoff for session in allowed})
        self.assertEqual(training["feature_sessions"], [sessions[0]])
        self.assertEqual(evaluation["feature_sessions"], [sessions[-1]])
        self.assertEqual([call.kwargs for call in adjust.call_args_list], [
            {"fields": ("open", "close"), "anchor_session": allowed[-1],
             "decision_session": allowed[-1], "factor_field": "factor"}
            for allowed in (sessions[:1], sessions, sessions[:1], sessions)])

    def test_label_only_helper_rejects_empty_training_scope(self):
        class NoReads:
            def read(self, **kwargs): raise AssertionError("empty scope queried Data")
        with self.assertRaisesRegex(ValueError, "training feature sessions required"):
            _prepare_stock_labels(NoReads(), config={}, training_sessions=[])


if __name__ == "__main__":
    unittest.main()
