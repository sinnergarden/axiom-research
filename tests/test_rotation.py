"""Independent boundary cases for the saved deterministic ETF baseline."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from axiom_data import DataBatch, QuerySpec
from axiom_research import ArtifactRef, SessionRange, semantic_identity
from axiom_research.data_adapter import AdapterError
from axiom_research.rotation import build_rotation_experiment, load_rotation_experiment
from axiom_research.rotation import _publish

DAYS = [(date(2026, 6, 1) + timedelta(days=i)).isoformat() for i in range(45)
        if (date(2026, 6, 1) + timedelta(days=i)).weekday() < 5]
SYMBOLS = ('A', 'B')
CUTOFFS = {s: s + 'T20:30:00+08:00' for s in DAYS}
PRICE = QuerySpec('market_daily', ('close',), SYMBOLS, tuple(DAYS), 'best_effort_vendor_v1', CUTOFFS)
FACTOR = replace(PRICE, domain='adjustment_factors', fields=('factor',))
PACKAGE = ArtifactRef(artifact_type='ImplementationPackage', artifact_id='synthetic-rotation-v1',
    artifact_contract_version='1', content_digest='sha256:'+'b'*64, uri='fixture:reviewed-package')


class Store:
    def load_snapshot(self, snapshot):
        return dict(snapshot_id=snapshot, domains={
            'market_daily': {'contract': {'fields': {'close': {'dtype': 'float64', 'unit': 'CNY/fund unit'}}}},
            'adjustment_factors': {'contract': {'fields': {'factor': {'dtype': 'float64', 'unit': 'dimensionless'}}}},
            'trading_calendar': {'synthetic_calendar': DAYS}})


class Fixture:
    store = Store()
    def __init__(self, *, missing=None, zero=None, release=None, factor_step=False):
        self.missing=missing; self.zero=zero; self.release=release; self.factor_step=factor_step; self.calls=[]

    def read(self, *, snapshot, query):
        self.calls.append((query.domain, query.sessions, dict(query.cutoff_by_session)))
        field=query.fields[0]; rows=[]; meta=[]
        for symbol in query.symbols:
            for session in query.sessions:
                i=DAYS.index(session)
                value=(100.0 if self.factor_step else 100.0+i*(1 if symbol=='A' else 2)) if field=='close' else (
                    2.0 if self.factor_step and i>=10 else 1.0)
                if (field,session,symbol)==self.zero: value=0.0
                missing=(field,session,symbol)==self.missing
                available=(self.release if (field,session,symbol)==('factor',DAYS[-1],'A') and self.release
                           else session+'T20:00:00+08:00')
                invisible=pd.Timestamp(available)>pd.Timestamp(query.cutoff_by_session[session])
                if missing or invisible: value=None
                rows.append(dict(security_id=symbol,session=session,**{field:value}))
                meta.append(dict(security_id=symbol,session=session,usable_from=None if invisible else available,
                    first_observed_at='2026-10-04T00:00:00Z',revision_id='synthetic:'+field+session,
                    raw_batch_id='synthetic',availability_basis='synthetic',evidence_ref='fixture',
                    missing_reason='not_visible_at_cutoff' if invisible else 'source_missing' if missing else None))
        context=dict(contract_version='data_batch_v1',snapshot_id=snapshot,domain=query.domain,
            reader_version='synthetic-reader',query=dict(fields=list(query.fields),symbols=list(query.symbols),
                sessions=list(query.sessions),pit_policy=query.pit_policy,cutoff_by_session=dict(query.cutoff_by_session),
                purpose=query.purpose,price_basis=query.price_basis,adjustment_anchor=None,universe_id=None,policy_by_session=None),
            limitations=['synthetic only'])
        return DataBatch(pd.DataFrame(rows[::-1]),{field:dict(dtype='float64',
            unit='CNY/fund unit' if field=='close' else 'dimensionless',by_key=meta[::-1])},context)

    def states(self, *, snapshot, query):
        self.calls.append(('states',query.sessions,dict(query.cutoff_by_session)))
        rows=[dict(security_id=s,session=d,market_state='unknown_status' if d in DAYS else 'calendar_closed')
              for s in query.symbols for d in query.sessions]
        meta=[dict(security_id=r['security_id'],session=r['session'],
                   missing_reason='status_source_missing' if r['session'] in DAYS else None) for r in rows]
        return DataBatch(pd.DataFrame(rows),{'market_state':{'by_key':meta}}, {'synthetic':True})


def build(data, destination, **changes):
    args=dict(snapshot='synthetic-s1',price_query=PRICE,factor_query=FACTOR,
        evaluation_range=SessionRange(start=DAYS[21],end=DAYS[-1]),destination=destination,
        implementation_package=PACKAGE)
    args.update(changes)
    return build_rotation_experiment(data,**args)


class RotationTests(unittest.TestCase):
    def test_concurrent_same_identity_publication_reuses_complete_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'winner';identity={'fixture':'same-identity'}
            def competitor(stage): (stage/'payload.json').write_text('{}')
            def write(stage):
                competitor(stage)
                self.assertFalse(_publish(target,'synthetic_publication_v1',identity,competitor,('payload.json',)))
            self.assertTrue(_publish(target,'synthetic_publication_v1',identity,write,('payload.json',)))
            self.assertEqual((target/'payload.json').read_text(),'{}')

    def test_adjustment_independent_formula_and_cutoffs(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Fixture(factor_step=True); run=build(data,directory)
            frame=run.signal_frame()
            values={r['session']:r for r in frame['rows'] if r['security_id']=='A'}
            self.assertFalse(values[DAYS[19]]['valid'])
            self.assertAlmostEqual(values[DAYS[20]]['score'],1.0)
            self.assertAlmostEqual(values[DAYS[-1]]['score'],0.0)
            self.assertEqual(values[DAYS[20]]['knowledge_cutoff'],DAYS[20]+'T12:30:00Z')
            self.assertTrue(all(r['available_at']<=r['knowledge_cutoff'] for r in frame['rows']))
            for domain, sessions, cutoffs in data.calls:
                if domain!='states':
                    self.assertEqual(set(cutoffs.values()),{CUTOFFS[sessions[-1]]})
                    self.assertLessEqual(len(sessions),21)
            self.assertEqual(frame['contract_version'],'signal_frame_v1')
            self.assertTrue(all(r['source_refs'] for r in frame['rows']))
            expanded=run.feature_frames()
            self.assertTrue(all('source_bindings' in f and 'sources' in f['rows'][0] for f in expanded))
            self.assertTrue(all(set(source for sources in row['sources'] for source in sources) <=
                {binding['id'] for binding in f['source_bindings']} for f in expanded for row in f['rows']))

    def test_interior_gap_nonpositive_price_and_unavailable_factor_fail_closed(self):
        for change in ({'missing':('factor',DAYS[15],'A')},{'zero':('close',DAYS[15],'A')},
                       {'release':DAYS[-1]+'T21:00:00+08:00'}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                rows=build(Fixture(**change),directory).signal_frame()['rows']
                a=next(r for r in rows if r['security_id']=='A' and r['session']==DAYS[-1])
                b=next(r for r in rows if r['security_id']=='B' and r['session']==DAYS[-1])
                self.assertIsNone(a['score']);self.assertFalse(a['valid']);self.assertTrue(a['invalid_reason'])
                self.assertTrue(b['valid'])

    def test_cache_reproduction_stage_reuse_and_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            first=build(Fixture(),Path(directory)/'one')
            second=build(Fixture(),Path(directory)/'two')
            self.assertEqual(first.signal_frame(),second.signal_frame())
            files=list(first.path.parent.parent.rglob('*.*'))
            before={str(p.relative_to(first.path.parent.parent)):(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
            class NoReads(Fixture):
                def read(self,**kwargs):raise AssertionError('no facts on cache hit')
                states=read
            with patch('axiom_research.rotation.execute_feature_plan',side_effect=AssertionError('no Core on reuse')):
                reused=build(NoReads(),Path(directory)/'one')
                self.assertTrue(reused.reused and reused.feature_reused and reused.signal_reused)
                narrowed=build(NoReads(),Path(directory)/'one',evaluation_range=SessionRange(start=DAYS[22],end=DAYS[-1]))
                self.assertTrue(narrowed.feature_reused and narrowed.signal_reused)
                self.assertFalse(narrowed.reused)
                moved=build(NoReads(),Path(directory)/'one',implementation_package=replace(PACKAGE,uri='elsewhere',metadata={'note':'display'}))
                self.assertTrue(moved.reused)
            self.assertTrue(all((first.path.parent.parent/key).read_bytes()==value[0] and
                (first.path.parent.parent/key).stat().st_mtime_ns==value[1] for key,value in before.items()))
            shutil.copytree(first.path.parent.parent,Path(directory)/'relocated')
            restored=load_rotation_experiment(Path(directory)/'relocated'/'experiments'/first.path.name)
            self.assertEqual(restored.signal_frame(),first.signal_frame())

    def test_calendar_compression_warmup_and_replay_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            omitted=tuple(s for s in DAYS if s!=DAYS[10]);cutoffs={s:CUTOFFS[s] for s in omitted}
            with self.assertRaisesRegex(AdapterError,'omit/include'):
                build(Fixture(),directory,price_query=replace(PRICE,sessions=omitted,cutoff_by_session=cutoffs),
                    factor_query=replace(FACTOR,sessions=omitted,cutoff_by_session=cutoffs),
                    evaluation_range=SessionRange(start=DAYS[22],end=DAYS[-1]))
            with self.assertRaisesRegex(AdapterError,'warmup'):
                build(Fixture(),directory,evaluation_range=SessionRange(start=DAYS[20],end=DAYS[-1]))
            with self.assertRaisesRegex(AdapterError,'decision facts'):
                build(Fixture(),directory,price_query=replace(PRICE,purpose='market_replay'))

    def test_corruption_and_partial_write_preserve_old_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('pyarrow.parquet.write_table',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):build(Fixture(),directory)
            self.assertEqual(list((Path(directory)/'features').iterdir()),[])
            run=build(Fixture(),directory)
            saved=run.signal_path/'signal-frame.json';saved.write_bytes(b'corrupt')
            with self.assertRaisesRegex(AdapterError,'integrity'):build(Fixture(),directory)
            self.assertEqual(saved.read_bytes(),b'corrupt')


if __name__=='__main__':unittest.main()
