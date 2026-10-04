"""Batch-field references preserve cell clocks while bounding CS provenance."""
import copy
import unittest
from test_data_adapter import batch,REF
from axiom_research.data_adapter import adapt_decision_batch,_digest


class CompactSourceTests(unittest.TestCase):
    def test_compact_sources_preserve_numerical_and_time_semantics(self):
        p=batch('close',[10.,None,20.]);r=batch('is_member',[True]*3,membership=True)
        cell=adapt_decision_batch(p,reference=r,recipe_ref=REF)
        compact=adapt_decision_batch(p,reference=r,recipe_ref=REF,source_granularity='batch_field')
        left=cell.execute().to_dict()['rows'];right=compact.execute().to_dict()['rows']
        for a,b in zip(left,right):
            for k in ('values','availability','valid','reasons'):self.assertEqual(a[k],b[k])
        self.assertLess(len(compact.facts.to_dict()['sources']),len(cell.facts.to_dict()['sources']))
        refs={_digest(p.to_json()),_digest(r.to_json())}
        self.assertEqual({v['batch_ref'] for v in compact.source_evidence.values()},refs)
        self.assertEqual(sum(len(v['provenance_by_key']) for v in compact.source_evidence.values()),6)

    def test_metadata_change_rebinds_compact_source_without_forging_values(self):
        p=batch('close',[10.,11.,12.]);r=batch('is_member',[True]*3,membership=True)
        a=adapt_decision_batch(p,reference=r,recipe_ref=REF,source_granularity='batch_field')
        changed=copy.deepcopy(p);changed.wire['field_meta']['close']['by_key'][0]['revision_id']='r2'
        b=adapt_decision_batch(changed,reference=r,recipe_ref=REF,source_granularity='batch_field')
        self.assertNotEqual(a.facts.identity,b.facts.identity)
        self.assertEqual([x['values'] for x in a.execute().to_dict()['rows']],
                         [x['values'] for x in b.execute().to_dict()['rows']])


if __name__=='__main__':unittest.main()
