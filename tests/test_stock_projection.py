"""Keyed native projection equivalence, nulls and frozen-revision rejection."""
from copy import deepcopy
import unittest

from axiom_research.stock_artifacts import digest
from axiom_research.stock_ml import _project_qlib


def fixture():
    import pandas as pd
    import numpy as np
    from axiom_data import DataBatch
    rows=[{'security_id':s,'session':d,'close':100.000001+i/10,'amount_cny':100001+i}
          for i,(s,d) in enumerate((s,d) for d in ['2024-01-02','2024-01-03'] for s in ['A','B','C'])]
    rows[2]['close']=float('nan')
    frame=pd.DataFrame(rows).iloc[[5,0,3,2,1,4]].copy()
    keys=[(r['security_id'],r['session']) for r in rows]
    values={k:{f:None if f=='close' and i==2 else float(np.float32(rows[i][f]))
               for f in ['close','amount_cny']} for i,k in reversed(list(enumerate(keys)))}
    query={'fields':['close','amount_cny'],'symbols':['A','B','C'],'sessions':['2024-01-02','2024-01-03'],
           'cutoff_by_session':{d:d+'T20:30:00+08:00' for d in ['2024-01-02','2024-01-03']}}
    batch=DataBatch(frame,{'close':{'by_key':[]},'amount_cny':{'by_key':[]}},
                    {'query':query,'snapshot_id':'synthetic-fixed','domain':'market_daily'})
    return batch,values


def original_scalar(batch,values,view_ref):
    """Frozen e1e9049 scalar admission, independent of the new vector path."""
    import numpy as np
    from axiom_data import DataBatch
    wire=batch.to_json();frame=batch.frame.copy()
    for i,row in enumerate(wire['records']):
        key=row['security_id'],row['session']
        for field in wire['context']['query']['fields']:
            expected=row[field];value=values[key][field]
            if expected is None:
                if value is not None: raise ValueError('Qlib/Reader missingness mismatch')
            elif value is None or float(np.float32(expected))!=value:
                raise ValueError('Qlib/Reader revision or value mismatch')
            frame.at[frame.index[i],field]=value
    context={**wire['context'],'numeric_projection':{
        'contract_version':'research_qlib_native_projection_v1','view_ref':view_ref,
        'reader_batch_ref':digest(wire),'revision_admission':'exact_reader_float32_value',
        'dtype':'float32_values_promoted_to_float64'}}
    return DataBatch(frame,batch.field_meta,context)


class StockProjectionTests(unittest.TestCase):
    def test_integer_dtype_boundary_never_silently_changes_qlib_value(self):
        import numpy as np
        import pandas as pd
        from axiom_data import DataBatch
        for dtype in ('int32','uint32','int64'):
            with self.subTest(dtype=dtype):
                maximum=int(np.iinfo(dtype).max)
                frame=pd.DataFrame({'security_id':['A'],'session':['2024-01-02'],
                    'amount_cny':pd.Series([maximum],dtype=dtype)})
                batch=DataBatch(frame,{}, {'query':{'fields':['amount_cny'],
                    'symbols':['A'],'sessions':['2024-01-02']}})
                native={('A','2024-01-02'):{'amount_cny':float(np.float32(maximum))}}
                with self.assertRaisesRegex(ValueError,'lossy integer'):
                    _project_qlib(batch,native,'fixed-view')
                self.assertEqual(int(batch.frame.iloc[0]['amount_cny']),maximum)

    def test_keyed_shuffle_matches_original_bytes_and_preserves_reader(self):
        batch,values=fixture();before=batch.to_json()
        original=original_scalar(batch,values,'fixed-view').to_json()
        current=_project_qlib(batch,values,'fixed-view').to_json()
        self.assertEqual(current,original);self.assertEqual(digest(current),digest(original))
        self.assertEqual(batch.to_json(),before)
        self.assertIs(type(current['records'][0]['amount_cny']),int)
        self.assertEqual(current['field_meta'],before['field_meta'])
        self.assertEqual(current['context']['query']['cutoff_by_session'],before['context']['query']['cutoff_by_session'])

    def test_duplicate_or_missing_reader_grid_and_missing_native_key_rejected(self):
        from axiom_data import DataBatch
        import pandas as pd
        batch,values=fixture()
        for frame,reason in ((pd.concat([batch.frame,batch.frame.iloc[:1]]),'duplicate'),
                             (batch.frame.iloc[:-1],'complete Reader')):
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason):
                _project_qlib(DataBatch(frame,batch.field_meta,batch.context),values,'fixed-view')
        missing=deepcopy(values);missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError,'complete Qlib'):
            _project_qlib(batch,missing,'fixed-view')

    def test_null_nan_and_revision_counterexamples_match_original(self):
        batch,values=fixture()
        for mode,key in [('revision',('A','2024-01-02')),('present_null',('A','2024-01-02')),
                         ('missing_present',('C','2024-01-02')),('native_nan',('C','2024-01-02'))]:
            changed=deepcopy(values)
            changed[key]['close']={'revision':99.,'present_null':None,'missing_present':10.,'native_nan':float('nan')}[mode]
            for project in (original_scalar,_project_qlib):
                with self.subTest(mode=mode,project=project.__name__),self.assertRaises(ValueError):
                    project(batch,changed,'fixed-view')


if __name__=='__main__':unittest.main()
