"""Public Label-builder counterexamples for borrowed mutable Raw snapshots.

Only synthetic public batch wires are used; no Data root or executor is opened.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import digest, file_digest, write_json
from axiom_research.stock_matrix_reader import (
    VerifiedMatrixStore, _admit_raw_label, _raw_snapshot_reservation,
)
from axiom_research.stock_native_json import make_native_carrier
from test_stock_labels import build, batch, CALENDAR


LIMIT=32*1024**2


class RawLeaseIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='axiom-raw-snapshot-')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.store=VerifiedMatrixStore(maximum_source_bytes=LIMIT,maximum_matrix_bytes=LIMIT)
        self.addCleanup(lambda: None if self.store.closed else self.store.close())

    def save(self,name,value,*,label_ref=None):
        path=self.root/name
        write_json(path,value)
        return {'path':str(path),'file_digest':file_digest(path),
                'label_ref':value['label_ref'] if label_ref is None else label_ref}

    def check_original(self,value,original):
        self.assertEqual(value,original)
        self.assertEqual(digest({k:v for k,v in value.items() if k!='label_ref'}),value['label_ref'])

    def test_public_raw_two_cached_paths_and_two_active_leases_are_isolated(self):
        original=build()
        self.assertEqual(original['rows'][0]['return'],0.5)
        a=self.save('a.json',original); b=self.save('b.json',original)
        cached_a=self.store.read_json(a,'label_ref')
        cached_b=self.store.read_json(b,'label_ref')
        self.assertIs(cached_a,cached_b)  # Reproduce the shared native-ref cache.
        before={d['path']:(file_digest(d['path']),Path(d['path']).stat().st_mtime_ns) for d in (a,b)}
        with _admit_raw_label(a,store=self.store) as first:
            self.assertIsNot(first.raw,cached_a)
            first.raw['rows'][0]['return']=0.75
            first.raw['source_evidence']['context']['query']['purpose']='caller mutation'
            self.check_original(cached_a,original)
            self.check_original(self.store.read_json(b,'label_ref'),original)
            with _admit_raw_label(b,store=self.store) as second:
                self.assertIsNot(first.raw,second.raw)
                self.assertEqual(self.store.borrowers,2)
                self.check_original(second.raw,original)
                second.raw['rows'].clear()
                self.assertEqual(first.raw['rows'][0]['return'],0.75)
        self.assertEqual(self.store.borrowers,0)
        self.assertEqual(self.store.lease_bytes,0)
        self.assertEqual(before,{d['path']:(file_digest(d['path']),Path(d['path']).stat().st_mtime_ns) for d in (a,b)})
        with _admit_raw_label(b,store=self.store) as again: self.check_original(again.raw,original)

    def test_inline_and_bundle_cached_child_alias_is_isolated(self):
        original=build()
        other=build(features=(CALENDAR[1],),horizon=3)
        body={'contract_version':'stock_label_bundle_v1','training':original,'evaluation':other}
        bundle={**body,'label_ref':digest(body)}
        a=self.save('inline.json',original)
        b=self.save('bundle.json',bundle,label_ref=original['label_ref'])
        cached=self.store.read_json(a,'label_ref')
        child=self.store.read_json(b,'label_ref',_raw_bundle=True)
        cached_bundle=self.store._decoded[b['path']]
        self.assertIs(cached,child)
        with _admit_raw_label(a,store=self.store) as first:
            first.raw['rows'][0]['return']=0.75
            self.check_original(child,original)
            with _admit_raw_label(b,store=self.store) as second: self.check_original(second.raw,original)
        self.check_original(cached_bundle['training'],original)

    def test_carrier_snapshot_does_not_mutate_cached_skeleton_or_binding(self):
        source=batch(); source.wire['context']['coverage']={'observed':[1,None,'甲']}
        original=build(source)
        a=make_native_carrier(self.root,original,'label_ref',maximum_source_bytes=LIMIT,
            maximum_parent_bytes=64*1024**2,maximum_workspace_bytes=LIMIT)
        cached=self.store.read_json(a,'label_ref')
        expected=deepcopy(cached)
        with _admit_raw_label(a,store=self.store) as first:
            first.raw['rows'][0]['return']=0.75
            first.storage_binding['carrier']['label_ref']='caller mutation'
            self.assertEqual(cached,expected)
            with _admit_raw_label(a,store=self.store) as second:
                self.assertEqual(second.raw,expected)
                self.assertEqual(second.storage_binding['carrier'],a)
                self.assertEqual(second.raw['rows'],original['rows'])
        self.assertEqual(self.store.metrics['coverage_validation_calls'],1)

    def test_copy_budget_rejects_before_deepcopy_without_borrower_leak(self):
        original=build(); a=self.save('budget.json',original)
        raw=self.store.read_json(a,'label_ref')
        required=self.store._native_caller_bytes(raw)+_raw_snapshot_reservation(raw,self.store)
        self.store.maximum_matrix_bytes=required-1
        from axiom_research import stock_matrix_reader as reader
        with patch.object(reader,'deepcopy',side_effect=AssertionError('copy before admission')) as copier:
            with self.assertRaisesRegex(ValueError,'snapshot copy workspace budget'):
                _admit_raw_label(a,store=self.store)
            copier.assert_not_called()
        self.assertEqual(self.store.borrowers,0)
        self.assertEqual(self.store.lease_bytes,0)
        self.assertFalse(self.store.closed)

    def test_dynamic_caller_is_included_in_snapshot_admission(self):
        original=build(); a=self.save('caller.json',original)
        self.store.read_json(a,'label_ref')
        self.store._caller_retained_bytes=lambda:LIMIT
        with self.assertRaisesRegex(ValueError,'workspace budget'):
            _admit_raw_label(a,store=self.store)
        self.assertEqual(self.store.borrowers,0)
        self.assertEqual(self.store.lease_bytes,0)


if __name__=='__main__': unittest.main()
