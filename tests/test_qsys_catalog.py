"""Source migration/identity checks and hand-computed Core recipe evidence."""
from datetime import date, timedelta
import math
import unittest
from importlib.resources import files

from axiom_engine.core import ABI, SEMANTICS, FeaturePlan, FactBatch, ExecutionContext, execute_feature_plan
from axiom_research.a01_catalog import load_a01_feature_catalog
from axiom_research.feature_catalog import build_feature_plan, load_feature_catalog, load_feature_catalog_ref, FeatureCatalogError, FeatureCatalog
from axiom_research.qsys_catalog import load_qsys_feature_catalog, load_qsys_feature_registry, select_qsys_features, qsys_daily_fields, qsys_input_requirements, qsys_financial_fields

REF = 'sha256:' + 'a' * 64


def fixture(count=760, changes=None):
    catalog = load_qsys_feature_catalog()
    nodes = catalog.to_dict()['shared_nodes'] + [n for f in catalog.definitions for n in f['core_nodes']]
    named = {n['name'] for n in nodes}
    leaves = sorted({a for n in nodes for a in n['inputs'] if a not in named})
    from axiom_research.qsys_input_contracts import CORE_INPUT_CONTRACTS
    def unit(n):return CORE_INPUT_CONTRACTS[n]['unit']
    schema = [dict(name=n,dtype='float64',unit=unit(n),stage='fact',missing='preserve') for n in leaves]
    sessions = [(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(count)]
    sources = [dict(id='synthetic',data_ref=REF,view_ref=REF,revision_policy='frozen-synthetic',qualification='synthetic',availability_basis='visible-cutoff')]
    rows=[]; keys=[]; reference=[]
    for security in ('A','B'):
        for i,s in enumerate(sessions):
            close = 100+i if security=='A' else 120+i*.5
            values = {n:1000.+i for n in leaves}
            values.update(open=close-1,high=close+3,low=close-2,close=close,vwap=close+.25,volume_shares=1000.+i*10,amount_cny=100000.+i*100,
                          pe=10.+i*.01,pb=2.+i*.001,total_market_value_cny=1e9,float_market_value_cny=5e8,net_inflow_cny=1000.,extra_large_buy_cny=5000.,extra_large_sell_cny=2000.,turnover_rate=2.+i*.001)
            if changes:values.update(changes.get((security,i),{}))
            keys.append([security,s]);reference.append(dict(security_id=security,session=s,member=True,industry=None,available_at=s+'T00:00:00Z',source='synthetic'))
            rows.append(dict(security_id=security,session=s,values=[values[n] for n in leaves],availability=[s+'T12:00:00Z']*len(leaves),sources=[['synthetic'] for n in leaves],missing_reasons=['SOURCE_MISSING' if values[n] is None else None for n in leaves]))
    base=FeaturePlan.from_dict(dict(abi=ABI,semantics=SEMANTICS,recipe_ref=REF,calendar_ref=REF,reference_ref=REF,reference_members={s:{'A':None,'B':None} for s in sessions},input_schema=schema,event_schema={},sources=sources,observation_domain='sessions',history_policy='partial',nodes=[],outputs=[],obligations=[]))
    facts=FactBatch.from_dict(dict(abi=ABI,calendar_ref=REF,schema=schema,sources=sources,rows=rows,event_schema={},events=[]))
    context=ExecutionContext.from_dict(dict(abi=ABI,calendar_ref=REF,reference_ref=REF,sessions=sessions,cutoffs={s:s+'T20:00:00Z' for s in sessions},history_keys=keys,output_keys=[[sec,sessions[-1]] for sec in ('A','B')],reference=reference))
    return base,facts,context


class QsysCatalogTests(unittest.TestCase):
    def test_388_traceable_rows_and_no_blocked_placeholders(self):
        registry=load_qsys_feature_registry();rows=registry['rows']
        self.assertEqual(len(rows),388)
        self.assertEqual(sum(r['status']=='executable' for r in rows),361)
        self.assertEqual(sum(r['status']=='alias' for r in rows),2)
        self.assertEqual(sum(r['status']=='blocked' for r in rows),25)
        for r in rows:
            self.assertTrue(r['expression']);self.assertTrue(r['qsys_source'])
            if r['status']=='blocked':self.assertNotIn('selection',r)
        for name in ('industry_ret_20d','margin_eligible','days_since_252d_high'):
            with self.subTest(name=name),self.assertRaisesRegex(FeatureCatalogError,'blocked'):
                select_qsys_features([name])
        self.assertEqual(select_qsys_features(['($close - $open) / $open'])[0].id,'ALP001')
        with self.assertRaisesRegex(FeatureCatalogError,'duplicate'):
            select_qsys_features(['KMID','($close - $open) / $open'])

    def test_all_original_definitions_and_nodes_are_unchanged(self):
        catalog=load_qsys_feature_catalog();actual={f['id']:f for f in catalog.definitions}
        nodes={n['name']:n for n in catalog.to_dict()['shared_nodes']}
        for old in (load_a01_feature_catalog(),load_feature_catalog(),FeatureCatalog(files("axiom_research").joinpath("catalogs","qsys_stock_v1.json").read_text())):
            for f in old.definitions:self.assertEqual(actual[f['id']],f)
            for n in old.to_dict()['shared_nodes']:self.assertEqual(nodes[n['name']],n)
            self.assertEqual(load_feature_catalog_ref(old.identity).identity,old.identity)
        self.assertEqual(load_feature_catalog_ref(catalog.identity).identity,catalog.identity)
        self.assertEqual(sum(f['group']=='ALP' for f in catalog.definitions),158)

    def test_old_semantic_refs_are_reusable_in_enlarged_catalog(self):
        from axiom_engine.feature_library import feature_definitions
        catalog=load_qsys_feature_catalog();base,_,_=fixture(2)
        semantics={c['name']:{'unit':c['unit'],'precision':'test-float64'} for c in base.to_dict()['input_schema']}
        for old in (load_a01_feature_catalog(),load_feature_catalog(),FeatureCatalog(files("axiom_research").joinpath("catalogs","qsys_stock_v1.json").read_text())):
            for normalized in (False,True):
                before=build_feature_plan(base,old.default_selection,catalog=old,normalized=normalized)
                after=build_feature_plan(base,old.default_selection,catalog=catalog,normalized=normalized)
                self.assertEqual(feature_definitions(before,input_semantics=semantics),feature_definitions(after,input_semantics=semantics))

    def test_tied_rank_and_exact_extension_contract(self):
        base,facts,context=fixture(5,{('A',i):{'close':v} for i,v in enumerate([1.,2.,2.,4.,2.])})
        catalog=load_qsys_feature_catalog()
        plan=build_feature_plan(base,select_qsys_features(['RANK5']),catalog=catalog)
        self.assertAlmostEqual(execute_feature_plan(plan,facts,context).to_dict()['rows'][0]['values'][0],.6)
        self.assertEqual(qsys_daily_fields(['VWAP0']),{'volume_shares':{'domain':'market_daily','dtype':'int64','unit':'shares'}})
        self.assertEqual(qsys_daily_fields(['$pe']),{'pe':{'domain':'valuation_daily','dtype':'float64','unit':'ratio'}})
        self.assertEqual(qsys_daily_fields(['$big_inflow']),{
            'extra_large_buy_cny':{'domain':'money_flow_daily','dtype':'float64','unit':'CNY'},
            'extra_large_sell_cny':{'domain':'money_flow_daily','dtype':'float64','unit':'CNY'}})
        self.assertEqual(qsys_daily_fields(['KMID']),{})

    def test_all_current_dependencies_match_data_and_adapter_contracts(self):
        from axiom_data.sources import TUSHARE_DAILY_CONTRACT
        from axiom_data.stock_input_sources import stock_input_source_spec
        from axiom_research.stock_feature_columns import FeatureColumnSource
        from axiom_research.qsys_input_contracts import DAILY_FIELD_CONTRACTS, CORE_INPUT_CONTRACTS
        canonical={'market_daily':TUSHARE_DAILY_CONTRACT}
        for endpoint in ('daily_basic','moneyflow','margin_detail','income','balancesheet','fina_indicator','stk_holdernumber','forecast'):
            spec=stock_input_source_spec(endpoint,identity_map={'000001.SZ':'A'},
                next_open_session_by_date={'2020-01-01':'2020-01-02'})
            canonical[spec['domain']]=spec['contract']
        from axiom_data.event_sources import CONTRACTS
        canonical['cash_flow_events']=CONTRACTS['cashflow']
        canonical['price_limits']=CONTRACTS['stk_limit']
        canonical['top_holders_reports']=CONTRACTS['top10_holders']
        from axiom_data.provider_local import CONTRACTS as LOCAL_CONTRACTS
        canonical['security_status']=LOCAL_CONTRACTS['suspend_d']
        # Covers even supported future fields such as ps; there is no CNY fallback.
        for field,spec in DAILY_FIELD_CONTRACTS.items():
            physical=canonical[spec['domain']]['fields'][field]
            self.assertEqual({k:physical.get(k) for k in ('dtype','unit')}, {k:spec[k] for k in ('dtype','unit')})
        self.assertEqual(DAILY_FIELD_CONTRACTS['ps']['unit'],'ratio')
        catalog=load_qsys_feature_catalog()
        self.assertNotIn('valuation_daily.ps',{d for f in catalog.definitions for d in f['dependencies']})
        requirements=qsys_input_requirements()
        extensions=requirements['daily_fields']
        if requirements['benchmark_fields']:
            extensions['benchmark_close']={'domain':'materialized_benchmark_daily','dtype':'float64','unit':'index points'}
        for bundle in requirements['financial_fields'].values():
            physical=canonical[bundle['domain']]['fields'][bundle['field']]
            if physical['dtype'] not in ('date','string'):
                self.assertEqual((physical['unit'],physical.get('basis','reported')),(bundle['unit'],bundle['basis']))
            self.assertEqual(bundle['visibility'],'strict_before_session_start')
            for leaf in bundle['outputs']:
                extensions[leaf]={'domain':'materialized_financial_events','dtype':'float64','unit':bundle['unit']}
        source=FeatureColumnSource({'spec':{'read_sessions':[],'universe':[]},'daily_fields':extensions},{})
        schema={c['name']:c for c in source.input_schema()}
        nodes=catalog.to_dict()['shared_nodes']+[n for f in catalog.definitions for n in f['core_nodes']]
        names={n['name'] for n in nodes}
        leaves={a for n in nodes for a in n['inputs'] if a not in names}
        for field in leaves:
            self.assertEqual({k:schema[field][k] for k in ('dtype','unit')},
                {k:CORE_INPUT_CONTRACTS[field][k] for k in ('dtype','unit')})
        for dep in {d for f in catalog.definitions for d in f['dependencies']}:
            domain,field=dep.split('.')
            if field in ('announcement_date','forecast_type'):
                self.assertIn(domain,canonical)
                continue
            if domain=='security_status':
                self.assertEqual(canonical[domain]['fields'][field]['dtype'],'bool')
                continue
            if domain=='benchmark_daily':
                self.assertEqual(CORE_INPUT_CONTRACTS['benchmark_close']['unit'],'index points')
                continue
            self.assertEqual(CORE_INPUT_CONTRACTS[field]['domain'],domain)
            if field=='vwap' or domain=='adjustment_factors':continue
            self.assertEqual(CORE_INPUT_CONTRACTS[field]['unit'],canonical[domain]['fields'][field]['unit'])

    def test_ttm_yoy_recipe_and_explicit_materialization_requirements(self):
        catalog=load_qsys_feature_catalog()
        selection=select_qsys_features(['ttm_revenue_yoy'])
        definition=catalog.select(selection)[0]
        self.assertEqual((definition['id'],definition['lookback']),('GRC004',1))
        self.assertEqual(definition['dependencies'],['financial_events.revenue'])
        self.assertEqual(qsys_input_requirements(['ttm_revenue_yoy'])['daily_fields'],{})
        bundle=qsys_financial_fields(['ttm_revenue_yoy'])['revenue_ttm_yoy']
        self.assertEqual(bundle['outputs'],{'ttm_revenue':{'derivation':'ttm','quarters_back':0},
            'ttm_revenue_prior_year':{'derivation':'ttm','quarters_back':4}})
        for names in (None,['ttm_revenue_yoy']):
            with self.assertRaisesRegex(FeatureCatalogError,'qsys_input_requirements'):
                qsys_daily_fields(names)
        for current,previous,expected in ((300.,200.,.5),(100.,-100.,-2.),(100.,0.,None),(None,200.,None),(100.,None,None)):
            base,facts,context=fixture(2,{('A',1):{'ttm_revenue':current,'ttm_revenue_prior_year':previous}})
            plan=build_feature_plan(base,selection,catalog=catalog)
            value=execute_feature_plan(plan,facts,context).to_dict()['rows'][0]['values'][0]
            if expected is None:self.assertIsNone(value)
            else:self.assertAlmostEqual(value,expected)
        self.assertEqual(select_qsys_features(['is_profitable_ttm'])[0].semantic_version,'2.0.0')

    def test_every_recipe_compiles_and_executes_through_core(self):
        catalog=load_qsys_feature_catalog();base,facts,context=fixture()
        for normalized in (False,True):
            plan=build_feature_plan(base,catalog.default_selection,catalog=catalog,normalized=normalized)
            self.assertEqual(len(plan.to_dict()['outputs']),378)
        plan=build_feature_plan(base,catalog.default_selection,catalog=catalog)
        out=execute_feature_plan(plan,facts,context).to_dict()
        actual={f['name']:v for f,v in zip(catalog.definitions,out['rows'][0]['values'])}
        self.assertTrue({
            'volume_up_down_ratio_60d','return_drawdown_ratio_60d','return_drawdown_ratio_120d',
            'up_volume_down_volume_ratio_60d','up_volume_down_volume_ratio_120d'} <= {name for name,value in actual.items() if value is None})
        close=859.; volume=8590.
        expected={'VWAP0':859.25/859,'BETA5':1/859,'RSQR5':1.,'RESI5':0.,'QTLU5':858.2/859,'QTLD5':855.8/859,'IMAX5':1.,'IMIN5':.2,'IMXD5':.8,
                  'VMA5':8570/(volume+1e-12),'VSTD5':math.sqrt(250)/(volume+1e-12),'VSUMP5':50/(50+1e-12),'VSUMN5':0.,'VSUMD5':50/(50+1e-12),
                  '$pe':17.59,'$big_inflow':3000.,'close_to_open_gap_1d':0.,'open_to_close_ret':859/858-1,'upper_shadow_ratio':.6,
                  'margin_balance_to_float_mv':1759/5e8,'volume_shock_3':8590/8580,'price_percentile_252d':1.,'pe_percentile_756d':1.}
        for name,value in expected.items():
            with self.subTest(name=name):self.assertAlmostEqual(actual[name],value,places=10)
        self.assertTrue(all(v is None or math.isfinite(v) for r in out['rows'] for v in r['values']))
        # Large-window recipes actually receive sufficient input in this fixture.
        self.assertIsNotNone(actual['pb_repair_room_to_median'])

    def test_subset_pruning_missing_and_normalization_boundary(self):
        catalog=load_qsys_feature_catalog();base,facts,context=fixture(12,{('A',10):{'volume_shares':None}})
        chosen=select_qsys_features(['VMA5','BETA5','Log($total_mv+1)'])
        plan=build_feature_plan(base,chosen,catalog=catalog)
        out=execute_feature_plan(plan,facts,context).to_dict()
        self.assertIsNone(out['rows'][0]['values'][0]);self.assertIsNotNone(out['rows'][0]['values'][1])
        self.assertAlmostEqual(out['rows'][0]['values'][2],math.log(1e9+1))
        normalized=build_feature_plan(base,chosen,catalog=catalog,normalized=True)
        self.assertNotEqual(plan.to_dict()['recipe_ref'],normalized.to_dict()['recipe_ref'])
        self.assertTrue(all(o['column']['stage']=='cross_sectional' for o in normalized.to_dict()['outputs']))
        self.assertNotIn('margin_balance',{a for n in plan.to_dict()['nodes'] for a in n['inputs']})


if __name__ == '__main__':unittest.main()
