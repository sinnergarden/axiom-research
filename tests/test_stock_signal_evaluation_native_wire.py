"""Captured saved wire shape; one semantic field changes in each rejection."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from axiom_research.stock_artifacts import digest
from axiom_research.stock_compact_labels import _write
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_signal_evaluation_inputs import _label_source_context
from axiom_research.stock_signal_evaluation_compact import _header, _spec
from axiom_research.stock_signal_evaluation_matrix import _target
from stock_signal_native_wire_fixture import SHAPE, saved_context


def change(value, path, replacement, *, remove=False):
    parent = value
    for name in path[:-1]: parent = parent[name]
    if remove: del parent[path[-1]]
    else: parent[path[-1]] = replacement


class CanonicalNativeWireTests(unittest.TestCase):
    def setUp(self):
        self.calendar = [f'2024-01-{day:02}' for day in range(2,10)]
        self.universe = ['S000']; self.sessions = [self.calendar[0]]
        self.cutoff = '2024-01-09T20:30:00+08:00'
        self.common = dict(snapshot='synthetic_fixed_snapshot',pit_policy='synthetic_pit',
                           calendar=self.calendar,universe=self.universe)
        self.scope = dict(calendar=self.calendar,universe=self.universe,sessions=self.sessions,
                          evaluation_cutoff=self.cutoff)
        self.context = saved_context(snapshot=self.common['snapshot'],pit=self.common['pit_policy'],
            universe=self.universe,sessions=[self.calendar[1],self.calendar[5],self.calendar[-1]],
            anchor=self.calendar[-1],cutoff=self.cutoff)

    def admit_context(self, context):
        return _label_source_context(context,self.scope,self.common['snapshot'],self.common['pit_policy'],
            price_basis='common_anchor_adjusted_v1',adjustment_anchor=self.calendar[-1],compact=True)

    def test_canonical_domain_locations_and_optional_native_consistency(self):
        universe,cutoff = self.admit_context(self.context)
        self.assertEqual(universe,self.universe)
        self.assertEqual(cutoff,datetime.fromisoformat(self.cutoff))
        self.assertNotIn('contract_version',self.context)
        self.assertNotIn('domain',self.context['derivation']['price_query'])
        self.assertNotIn('domain',self.context['derivation']['factor_query'])
        for name,domain in [('price_query','market_daily'),('factor_query','adjustment_factors')]:
            context = deepcopy(self.context);context['derivation'][name]['domain'] = domain
            self.admit_context(context)

    def test_context_single_field_rejections(self):
        cases = [
            ('missing_price_domain',('domain',),None,True),
            ('wrong_price_domain',('domain',),'adjustment_factors',False),
            ('missing_factor_domain',('derivation','factor_domain'),None,True),
            ('wrong_factor_domain',('derivation','factor_domain'),'market_daily',False),
            ('wrong_optional_price_domain',('derivation','price_query','domain'),'adjustment_factors',False),
            ('wrong_optional_factor_domain',('derivation','factor_query','domain'),'market_daily',False),
            ('snapshot',('snapshot_id',),'other_snapshot',False),
            ('pit',('query','pit_policy'),'other_pit',False),
            ('universe',('query','symbols'),['other_security'],False),
            ('purpose',('query','purpose'),'decision_facts',False),
            ('fields',('query','fields'),['open'],False),
            ('basis',('query','price_basis'),'unadjusted',False),
            ('cutoff_keys',('query','cutoff_by_session'),{},False),
            ('query_cutoff',('query','cutoff_by_session',self.calendar[1]),'2025-01-01T00:00:00Z',False),
            ('calendar',('query','sessions'),['2025-01-01'],False),
            ('recipe',('derivation','recipe_version'),'other_recipe',False),
            ('formula',('derivation','formula'),'other_formula',False),
            ('decision_session',('derivation','decision_session'),self.calendar[0],False),
            ('anchor',('derivation','anchor_session'),self.calendar[0],False),
            ('adjustment_anchor',('query','adjustment_anchor'),self.calendar[0],False),
            ('decision_cutoff',('derivation','decision_cutoff'),'2024-01-09T19:30:00+08:00',False),
            ('factor_field',('derivation','factor_field'),'other_factor',False),
        ]
        for name in ('price_query','factor_query'):
            for field,replacement in [
                ('pit_policy','other_pit'),('purpose','decision_facts'),('price_basis','adjusted'),
                ('adjustment_anchor',self.calendar[0]),('symbols',['other_security']),
                ('sessions',[self.calendar[-1]]),('fields',[]),('cutoff_by_session',{}),
                ('policy_by_session',dict.fromkeys(self.context['query']['sessions'],'other_pit'))]:
                cases.append((name+'_'+field,('derivation',name,field),replacement,False))
        for label,path,value,remove in cases:
            with self.subTest(field=label):
                context = deepcopy(self.context); change(context,path,value,remove=remove)
                with self.assertRaises((ValueError,KeyError)): self.admit_context(context)
        self.assertEqual(len(cases),40)

    def target(self,root):
        price = seal(dict(contract_version='stock_label_price_view_v1',context=self.context,
            records_ref=digest('synthetic prices'),field_meta_ref=digest('synthetic metadata')),'price_view_ref')
        definition = dict(calendar=self.calendar,cutoff=self.cutoff,end_session_offset=5,
            formula='close(f+5) / open(f+1) - 1',horizon_sessions=5,implementation_ref=digest('synthetic operator'),
            missing_policy='invalid_null_preserve_grid',price_basis='common_anchor_adjusted_v1',
            price_view=price,sessions=self.sessions,snapshot=self.common['snapshot'],start_session_offset=1,
            universe=self.universe)
        row = dict(security_id=self.universe[0],feature_session=self.sessions[0],
            start_session=self.calendar[1],end_session=self.calendar[5],return_=0.125,valid=True,
            invalid_reason=None,label_available_at=self.calendar[5]+'T12:00:00Z',source_refs=[price['price_view_ref']])
        row['return'] = row.pop('return_')
        # Use the actual owner's physical writer, never a hand-built header.
        descriptor = _write(root,definition,[row])
        import json
        header = json.loads(Path(descriptor['path']).read_text())
        records = {descriptor['path']:descriptor['file_digest'],
                   **{d['path']:d['file_digest'] for d in header['buffers'].values()}}
        return dict(descriptor=descriptor,header=header),records,row

    @staticmethod
    def reseal(source):
        # Only derived digests are refreshed after the one semantic mutation.
        header = source['header'];price = header['definition']['price_view']
        if 'price_view_ref' in price:
            price['price_view_ref'] = digest({k:v for k,v in price.items() if k!='price_view_ref'})
        header['definition_ref'] = digest(header['definition'])
        header['target_ref'] = digest({k:v for k,v in header.items() if k!='target_ref'})
        source['descriptor']['target_ref'] = header['target_ref']

    def test_owner_header_shape_versions_pins_and_single_field_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            source,records,row = self.target(Path(temp))
            header = source['header'];definition = header['definition']
            self.assertEqual(sorted(header),SHAPE['raw_header_fields'])
            self.assertEqual(sorted(definition),SHAPE['raw_definition_fields'])
            self.assertEqual(_header(source,self.scope,self.common,records),_spec(definition))
            _target(row,(row['security_id'],row['feature_session']),self.calendar,datetime.fromisoformat(self.cutoff),5)
            cases = [
                ('version',('contract_version',),'stock_compact_normalized_v1',False),
                ('missing_field',('cohort',),None,True),
                ('snapshot',('definition','snapshot'),'other_snapshot',False),
                ('universe',('definition','universe'),['other_security'],False),
                ('calendar',('definition','calendar'),self.calendar[:-1],False),
                ('sessions',('definition','sessions'),[],False),
                ('count',('row_count',),2,False),
                ('core_ref',('core_ref',),digest('other_core'),False),
                ('cohort',('cohort',),{},False),
                ('missing_buffer',('buffers','source_codes'),None,True),
                ('dtype',('buffers','values','dtype'),'int64_le',False),
                ('shape',('buffers','values','shape'),[2],False),
                ('float_shape',('buffers','values','shape'),[1.0],False),
                ('buffer_escape',('buffers','values','path'),'/outside/value.bin',False),
                ('buffer_missing_digest',('buffers','values','buffer_digest'),None,True),
                ('buffer_digest',('buffers','values','buffer_digest'),digest('other_buffer'),False),
                ('price_version',('definition','price_view','contract_version'),'other_version',False),
                ('price_records_ref',('definition','price_view','records_ref'),'invalid_ref',False),
                ('price_meta_ref',('definition','price_view','field_meta_ref'),'invalid_ref',False),
                ('source_snapshot',('definition','price_view','context','snapshot_id'),'other_snapshot',False),
                ('future_cutoff',('definition','cutoff'),'2025-01-01T00:00:00Z',False),
                ('vintage_cutoff',('definition','cutoff'),'2024-01-09T19:30:00+08:00',False),
                ('horizon',('definition','horizon_sessions'),4,False),
                ('start_offset',('definition','start_session_offset'),2,False),
                ('end_offset',('definition','end_session_offset'),4,False),
                ('formula',('definition','formula'),'other_formula',False),
                ('missing_policy',('definition','missing_policy'),'drop',False),
            ]
            for label,path,value,remove in cases:
                with self.subTest(field=label):
                    changed = deepcopy(source);change(changed['header'],path,value,remove=remove);self.reseal(changed)
                    with self.assertRaises((ValueError,KeyError)): _header(changed,self.scope,self.common,records)
            for label,path,value in [('relative_path',('descriptor','path'),'relative/target.json'),
                    ('header_pin',('descriptor','file_digest'),digest('wrong_target')),
                    ('target_ref',('descriptor','target_ref'),digest('wrong_target_ref'))]:
                with self.subTest(field=label):
                    changed = deepcopy(source);change(changed,path,value)
                    with self.assertRaises(ValueError): _header(changed,self.scope,self.common,records)
            bad_records = {**records,next(iter(header['buffers'].values()))['path']:digest('unbound_bytes')}
            with self.assertRaisesRegex(ValueError,'buffer pin'): _header(source,self.scope,self.common,bad_records)

    def test_actual_row_key_endpoint_source_value_reason_and_clock_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,row = self.target(Path(temp));key = row['security_id'],row['feature_session']
            for field,replacement in [('security_id','other_security'),('valid',1),('start_session',self.calendar[2]),
                    ('end_session',self.calendar[4]),('source_refs',[]),('return',None),('invalid_reason','unexpected'),
                    ('label_available_at',None),('label_available_at','2025-01-01T00:00:00Z')]:
                with self.subTest(field=field,value=replacement):
                    changed = deepcopy(row);changed[field] = replacement
                    with self.assertRaises(ValueError): _target(changed,key,self.calendar,datetime.fromisoformat(self.cutoff),5)
