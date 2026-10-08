"""Small 300-column saved-table acceptance; no Feature execution or real fit."""
from copy import deepcopy
import gc
from pathlib import Path
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

from axiom_research import (load_stock_feature_view,prepare_stock_ml_batch_inputs,
    load_stock_ml_batch_inputs,build_stock_ml_fold_from_saved_inputs,load_stock_ml_fold)
from axiom_research.stock_artifacts import _read,digest
from axiom_research.stock_fold_inputs import validate_spec
from axiom_research.stock_compact_store import (OwnedStore,_view_data,
    feature_rows,training_matrix,iter_feature_eligibility,set_feature_window,clear_feature_window)
from axiom_research.stock_compact_batch import read_target
import test_stock_matrix_feature_sources as feature_sources
from test_stock_matrix_prepare import PublicDataFixture,Query
from test_stock_folds import backend


class ModelFeatureSelectionTests(unittest.TestCase):
    options={'row_block_sessions':64,'column_block':32,'maximum_resident_bytes':256*1024**2,
        'normalization_backend':'core_cs_batch_v1'}

    def fixture(self,root):
        f=feature_sources.FoldSourceFixture(root)
        f.columns=['SYN'+str(i).zfill(3) for i in range(300)]
        f.selection=[{'id':c,'semantic_version':'1.0.0'} for c in f.columns]
        f.spec.update(ordered_features=f.columns,feature_selection=f.selection)
        original_day=f.day; original_catalog=f.catalog
        def wide(day):
            rows,proof=original_day(day)
            proof['core_plan']['outputs']=[{'node':c,'column':{'name':c,'dtype':'float64',
                'unit':'dimensionless','stage':'cross_sectional','missing':'preserve'}} for c in f.columns]
            for row in rows:
                row.update(values=[float(i+1) for i in range(300)],validity=[True]*300,
                    availability=[day+'T20:00:00.123456+08:00']*300,reasons=[[] for _ in f.columns],
                    source_refs=[proof['core_frame_ref'],digest(proof['core_plan'])])
                if row['security_id']=='A':
                    row['values'][299]=None; row['validity'][299]=False
                    row['availability'][299]=None; row['reasons'][299]=['SYNTHETIC_MISSING']
            return rows,proof
        f.day=wide
        # The existing synthetic producer supplies source proof; the production
        # v3 writer alone constructs metadata, partitions, identities and index.
        with f.matrix(options={'layout':'matrix_v2','row_block_sessions':90,'column_block':32,
                'maximum_resident_bytes':256*1024**2}) as saved:
            path=saved.path
            self.assertEqual(saved.to_dict()['contract_version'],'stock_feature_inputs_v3')
            self.assertTrue(saved.to_dict()['partitions'])
            self.assertEqual(len(saved.to_dict()['partitions'])%10,0)
        f.catalog=type('Catalog',(),{'identity':f.spec['catalog_ref'],
            'select':lambda _,selection:[{**deepcopy(original_catalog.select(f.selection)[0]),
                'id':s['id'],'semantic_version':s['semantic_version']} for s in selection]})()
        return f,path

    def raw_batch(self,f,path,root):
        data=PublicDataFixture(f.spec); module=types.ModuleType('axiom_data')
        module.QuerySpec=Query; module.adjust_prices=data.adjust
        with patch.dict(sys.modules,{'axiom_data':module}):
            return prepare_stock_ml_batch_inputs(data,feature_inputs=path,fold_specs=f.folds()[:1],
                destination=root/'raw-origin',preparation_options=self.options)

    def selection(self,f,start,stop): return deepcopy(f.selection[start:stop])

    def counts(self,view):
        m=_view_data(view)['store'].metrics
        return tuple(m.get(k,0) for k in ('model_column_part_admissions','file_hash_calls','json_decode_calls'))

    def test_four_parts_then_one_then_reorder_without_io_and_selected_qualification(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path=self.fixture(Path(temp)); a=self.selection(f,0,100); b=self.selection(f,50,150)
            with load_stock_feature_view(path,residency='sequential') as view:
                original=view.to_dict(); before=self.counts(view); offsets=[0,1,2]
                set_feature_window(view,offsets,model_feature_selection=a)
                after_a=self.counts(view); self.assertEqual(after_a[0]-before[0],4)
                self.assertEqual(after_a[2]-before[2],1)
                self.assertEqual(view.metrics['active_model_feature_blocks'],1)
                ra=feature_rows(view,offsets,model_feature_selection=a)
                self.assertTrue(all(ra[0]['validity'])); self.assertEqual(len(ra[0]['values']),100)
                set_feature_window(view,offsets,model_feature_selection=b)
                after_b=self.counts(view); self.assertEqual(after_b[0]-after_a[0],1)
                self.assertEqual(after_b[2],after_a[2])
                reverse=list(reversed(b)); rb=feature_rows(view,offsets,model_feature_selection=b)
                rr=feature_rows(view,offsets,model_feature_selection=reverse)
                self.assertEqual(self.counts(view),after_b)
                self.assertEqual(rr[0]['values'],list(reversed(rb[0]['values'])))
                self.assertEqual(view.to_dict(),original)
                facts=list(iter_feature_eligibility(view,offsets,model_feature_selection=a))
                self.assertTrue(facts[0].validity_all)
                selected_missing=self.selection(f,299,300)
                missing=list(iter_feature_eligibility(view,offsets,model_feature_selection=selected_missing))
                self.assertFalse(missing[0].validity_all); self.assertTrue(missing[1].validity_all)
                with self.assertRaisesRegex(ValueError,'membership/clock'):
                    training_matrix(view,[0],f.calendar[-1]+'T21:00:00+08:00',model_feature_selection=selected_missing)
                print('MODEL_COLUMN_IO '+str({'A_new_parts':4,'B_reused_parts':3,'B_new_parts':1,
                    'B_reorder_new_hashes':0,'shared_metadata_decodes':1,
                    'A_new_hashes':after_a[1]-before[1],'B_new_hashes':after_b[1]-after_a[1]}))
                # A retained exception must not retain credited physical arrays.
                import axiom_research.stock_compact_store as owner
                set_feature_window(view,[]); baseline=_view_data(view)['store'].resident_bytes
                array_refs=[]; original_validate=owner._validate_feature_cells
                def rejected(value,group,rows):
                    try:
                        array_refs.extend(weakref.ref(arrays['values']) for _,arrays in group)
                        original_validate(value,group,rows)
                        raise ValueError('synthetic post-admission failure')
                    finally: value=group=rows=None
                caught=None
                with patch.object(owner,'_validate_feature_cells',new=rejected):
                    try: set_feature_window(view,offsets,model_feature_selection=a)
                    except ValueError as exc: caught=exc
                self.assertIsNotNone(caught); gc.collect()
                self.assertTrue(all(ref() is None for ref in array_refs))
                state=_view_data(view)
                self.assertEqual(state['store'].arrays,{})
                self.assertTrue(all('model_parts' not in block for block in state['blocks']))
                self.assertEqual(view.metrics['active_model_feature_blocks'],0)
                self.assertEqual(state['store'].resident_bytes,baseline)
                set_feature_window(view,offsets,model_feature_selection=a)
                self.assertEqual(len(feature_rows(view,offsets,model_feature_selection=a)[0]['values']),100)

    def test_missing_duplicate_and_wrong_version_reject_before_column_io(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path=self.fixture(Path(temp))
            with load_stock_feature_view(path,residency='sequential') as view, \
                 patch.object(OwnedStore,'buffer',side_effect=AssertionError('column IO reached')):
                for selected in ([{'id':'absent','semantic_version':'1.0.0'}],
                        [f.selection[0],f.selection[0]],[{**f.selection[0],'semantic_version':'2.0.0'}]):
                    with self.subTest(selected=selected),self.assertRaises(ValueError):
                        feature_rows(view,[0],model_feature_selection=selected)
                    with self.assertRaises(ValueError):
                        prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],
                            destination=Path(temp)/'bad',preparation_options=self.options,model_feature_selection=selected)

    def test_retained_gather_failure_releases_full_and_selected_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path=self.fixture(Path(temp))
            for selection in (None,self.selection(f,0,100)):
                with self.subTest(selection='full' if selection is None else 'selected'), \
                     load_stock_feature_view(path,residency='sequential') as view:
                    options={} if selection is None else {'model_feature_selection':selection}
                    set_feature_window(view,[1],**options)
                    state=_view_data(view)
                    released_before=state['store'].metrics['released_buffer_bytes']
                    buffer_bytes=sum(len(payload) for payload in state['store'].arrays.values())
                    block=state['blocks'][state['day_blocks'][0]]
                    group=block['parts'] if selection is None else block['model_parts'].values()
                    refs=[weakref.ref(arrays['values']) for _,arrays in group]
                    block=group=None
                    def fail_gather(amount): raise ValueError('synthetic gather failure')
                    caught=None
                    with patch.object(state['store'],'reserve',new=fail_gather):
                        try: training_matrix(view,[1],f.calendar[-1]+'T21:00:00+08:00',**options)
                        except ValueError as exc: caught=exc
                    self.assertIsNotNone(caught)
                    clear_feature_window(view); gc.collect()
                    self.assertTrue(refs and all(ref() is None for ref in refs))
                    self.assertEqual(state['store'].arrays,{})
                    self.assertEqual(state['store'].metrics['released_buffer_bytes']-released_before,buffer_bytes)
                    matrix=training_matrix(view,[1],f.calendar[-1]+'T21:00:00+08:00',**options)
                    self.assertEqual(matrix.shape,(1,300 if selection is None else 100))

    def test_raw_reuse_requires_complete_fold_collection_before_admission(self):
        from axiom_research.stock_batch import _data
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); data=PublicDataFixture(f.spec)
            module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
            with patch.dict(sys.modules,{'axiom_data':module}):
                origin=prepare_stock_ml_batch_inputs(data,feature_inputs=path,fold_specs=f.folds()[:2],
                    destination=root/'origin-two',preparation_options=self.options)
            with load_stock_feature_view(path,residency='sequential') as view, \
                 load_stock_ml_batch_inputs(origin,feature_inputs=view,residency='sequential') as source, \
                 patch.object(_data(source)['matrix_state'],'_parts',side_effect=AssertionError('control admission reached')), \
                 patch.object(OwnedStore,'buffer',side_effect=AssertionError('column/target IO reached')), \
                 patch('axiom_research.stock_compact_labels._price_view',side_effect=AssertionError('Data reached')), \
                 patch('axiom_research.stock_compact_labels._normalized',side_effect=AssertionError('Core reached')):
                with self.assertRaisesRegex(ValueError,'fold/cutoff collection'):
                    prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],
                        destination=root/'single',preparation_options=self.options,
                        model_feature_selection=self.selection(f,0,100),reuse_raw_from_batch=source)
                self.assertFalse((root/'single').exists())

    def test_raw_reuse_changed_cohort_original_core_and_saved_model_bindings(self):
        from axiom_engine.core import FeaturePlan,FactBatch,ExecutionContext,execute_feature_plan
        from axiom_research.stock_label_contracts import normalization_section_inputs
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); origin=self.raw_batch(f,path,root)
            a=self.selection(f,0,100); b=self.selection(f,50,150)
            with load_stock_feature_view(path,residency='sequential') as view, \
                 load_stock_ml_batch_inputs(origin,feature_inputs=view,residency='sequential') as source:
                original=view.to_dict(); metrics={}
                origin_again=prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],
                    destination=root/'raw-origin',preparation_options=self.options,model_feature_selection=None)
                self.assertEqual(origin_again,origin)
                self.assertNotIn('model_feature_selection',origin['definition'])
                with patch('axiom_research.stock_ml._iter_stock_feature_days',side_effect=AssertionError('Feature executed')):
                    chosen=prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],
                        destination=root/'subsets',preparation_options=self.options,metrics=metrics,
                        model_feature_selection=a,reuse_raw_from_batch=source)
                self.assertEqual((metrics['data_read_calls'],metrics['feature_core_calls'],metrics['raw_operator_calls']),(0,0,0))
                self.assertEqual(metrics['label_core_calls'],1); self.assertEqual(view.to_dict(),original)
                with load_stock_ml_batch_inputs(chosen) as eager, \
                     load_stock_ml_batch_inputs(chosen,feature_inputs=view,residency='sequential') as sequential:
                    fold=chosen['folds'][0]
                    with eager._matrix_project(fold['input_manifest'],fold['fold_spec']) as left, \
                         sequential._matrix_project(fold['input_manifest'],fold['fold_spec']) as right:
                        for name in ('X','y','P'):
                            x,y=getattr(left,name),getattr(right,name)
                            self.assertEqual((x.shape,x.dtype,x.tobytes()),(y.shape,y.dtype,y.tobytes()))
                        self.assertEqual(left.training_keys,right.training_keys)
                        self.assertEqual(left.common,right.common)
                        self.assertEqual(left.features,right.features)
                        self.assertEqual(left.labels,right.labels)
                    with eager._project_evaluation(fold['input_manifest'],fold['fold_spec']) as left, \
                         sequential._project_evaluation(fold['input_manifest'],fold['fold_spec']) as right:
                        self.assertEqual(left.fold_binding,right.fold_binding)
                saved=chosen['folds'][0]; spec=saved['fold_spec']; inputs=saved['input_manifest']
                control=_read(inputs['fold_control']['path']); original_control=_read(origin['folds'][0]['input_manifest']['fold_control']['path'])
                self.assertEqual(control['raw_parts'],original_control['raw_parts'])
                self.assertEqual(control['evaluation_parts'],original_control['evaluation_parts'])
                norm=_read(control['normalized']['path']); old_norm=_read(original_control['normalized']['path'])
                self.assertNotEqual(norm['cohort']['eligible_keys'],old_norm['cohort']['eligible_keys'])
                self.assertTrue(any(key[0]=='A' for key in norm['cohort']['eligible_keys']))
                train,_=validate_spec(spec,f.calendar); positions={d:i for i,d in enumerate(f.days)}
                offsets=[positions[d]*len(f.universe)+j for d in train for j in range(len(f.universe))]
                feature_index={(r['security_id'],r['session']):r for r in feature_rows(view,offsets,model_feature_selection=a)}
                with OwnedStore() as store:
                    raw=[row for desc in control['raw_parts'] for row in read_target(store,desc)[1]]
                    normalized=read_target(store,control['normalized'],raw_rows=raw)[1]
                    rows={(r['security_id'],r['feature_session']):r for r in normalized}
                    raw_build={'label_ref':digest([d['target_ref'] for d in control['raw_parts']]),
                        'calendar_ref':digest({'contract_version':'stock_label_calendar_v1','sessions':f.calendar}),
                        'rows':raw,'label_spec':{'synthetic':'raw 5D oracle'}}
                    for day in train:
                        request=normalization_section_inputs(raw_build,feature_ref=view.identity,feature_rows=feature_index,
                            session=day,securities=f.universe,width=100,cutoff=spec['fit_cutoff'])
                        golden=execute_feature_plan(FeaturePlan.from_dict(request['plan']),FactBatch.from_dict(request['facts']),
                            ExecutionContext.from_dict(request['context'])).to_dict()
                        for row in golden['rows']:
                            actual=rows[row['security_id'],day]
                            self.assertEqual(actual['valid'],row['valid'][0])
                            self.assertEqual(actual['return'],row['values'][0])
                            if actual['valid']: self.assertEqual(actual['label_available_at'],row['availability'][0])
                before=self.counts(view)
                chosen_b=prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],
                    destination=root/'subsets',preparation_options=self.options,model_feature_selection=b,reuse_raw_from_batch=source)
                after=self.counts(view)
                retained=sum('model_rows' in block for block in _view_data(view)['blocks'])
                self.assertEqual(after[0]-before[0],retained)  # One new column part per retained row block.
                reversed_b=list(reversed(b)); reverse_metrics={}
                reordered=prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],
                    destination=root/'subsets',preparation_options=self.options,metrics=reverse_metrics,
                    model_feature_selection=reversed_b,reuse_raw_from_batch=source)
                self.assertEqual(self.counts(view),after); self.assertEqual(reverse_metrics['label_core_calls'],0)
                self.assertNotEqual(chosen_b['batch_ref'],reordered['batch_ref'])
                with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
                     patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend), \
                     patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
                     patch('axiom_research.stock_ml._environment',return_value=f.environment):
                    runs=[]
                    for batch_manifest,selection in ((chosen_b,b),(reordered,reversed_b)):
                        with load_stock_ml_batch_inputs(batch_manifest,feature_inputs=view,residency='sequential') as batch:
                            fold=batch_manifest['folds'][0]
                            run=build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                                destination=root/'models',batch=batch,model_feature_selection=selection)
                            runs.append(run)
                            model=_read(run.path/'model.json'); dataset=_read(run.path/'dataset.json')
                            self.assertEqual(model['ordered_features'],[s['id'] for s in selection])
                            self.assertEqual(model['model_feature_selection'],dataset['model_feature_selection'])
                            self.assertEqual(model['model_feature_selection']['feature_view_ref'],view.identity)
                            self.assertEqual(load_stock_ml_fold(run.path,batch=batch).identity,run.identity)
                            with patch('axiom_research.stock_compact_batch.training_matrix',side_effect=AssertionError('X reached')):
                                with self.assertRaisesRegex(ValueError,'prepared cohort'):
                                    build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                                        destination=root/'wrong',batch=batch,model_feature_selection=a)
                    self.assertNotEqual(runs[0].identity,runs[1].identity)
                    self.assertEqual(load_stock_ml_fold(runs[0].path).identity,runs[0].identity)
                missing_selected=prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],
                    destination=root/'subsets',preparation_options=self.options,
                    model_feature_selection=self.selection(f,299,300),reuse_raw_from_batch=source)
                missing_control=_read(missing_selected['folds'][0]['input_manifest']['fold_control']['path'])
                missing_norm=_read(missing_control['normalized']['path'])
                self.assertEqual(missing_norm['core_ref'],old_norm['core_ref'])
                self.assertEqual(missing_norm['cohort']['eligibility_reasons'],old_norm['cohort']['eligibility_reasons'])
                self.assertEqual(missing_norm['cohort']['eligible_keys'],old_norm['cohort']['eligible_keys'])
                with OwnedStore() as store:
                    for key in ('values','validity','availability','availability_validity','reason_codes'):
                        self.assertEqual(store.buffer(missing_norm['buffers'][key]).tobytes(),
                            store.buffer(old_norm['buffers'][key]).tobytes())
                # Reuse binds the original fold vintage and its query plan.
                changed=deepcopy(f.folds()[:1]); changed[0]['evaluation_cutoff']=f.calendar[-1]+'T20:31:00+08:00'
                with patch.object(OwnedStore,'buffer',side_effect=AssertionError('column/target IO reached')):
                    with self.assertRaisesRegex(ValueError,'fold/cutoff'):
                        prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=changed,destination=root/'bad-vintage',
                            preparation_options=self.options,model_feature_selection=a,reuse_raw_from_batch=source)
                    with self.assertRaisesRegex(ValueError,'row-block query plan'):
                        prepare_stock_ml_batch_inputs(None,feature_inputs=view,fold_specs=f.folds()[:1],destination=root/'bad-plan',
                            preparation_options={**self.options,'row_block_sessions':32},model_feature_selection=a,
                            reuse_raw_from_batch=source)


if __name__=='__main__': unittest.main()
