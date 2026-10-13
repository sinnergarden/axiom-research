"""Maintenance compiler for the fixed 852bcb QSYS inventory.

Input is the reviewed 388-row CSV; outputs are installed JSON definitions and
forensic register. This translates formulas to the existing Core ABI and does
not execute numerical calculations. Review source and tests before regenerating.
"""
import ast,csv,json,re,copy
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Compile reviewed QSYS migration recipes into Core DAG JSON; never executes data.')
parser.add_argument('inventory_csv', type=Path)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(args.inventory_csv.open()))
assert len(rows)==388 and [int(r['inventory_row']) for r in rows]==list(range(1,389))
cat=json.load(open(ROOT/'src/axiom_research/catalogs/a01_price_v1.json'))
legacy=json.load(open(ROOT/'src/axiom_research/catalogs/stock_ml_v1.json'))
cat['catalog_id']='qsys_stock_v3'
cat['shared_nodes']+=legacy['shared_nodes'];cat['features']+=legacy['features']
groups={g['id'] for g in cat['groups']}
for g in legacy['groups']:
 if g['id'] not in groups:cat['groups'].append(g);groups.add(g['id'])
units={'open':'CNY/share','high':'CNY/share','low':'CNY/share','close':'CNY/share','vwap':'CNY/share','volume_shares':'shares','amount_cny':'CNY','factor':'dimensionless','pe':'ratio','pb':'ratio','ps':'ratio','turnover_rate':'percent','total_mv':'CNY','circ_mv':'CNY','net_inflow':'CNY','big_inflow':'CNY','margin_balance':'CNY','margin_buy_amount':'CNY','margin_repay_amount':'CNY','margin_total_balance':'CNY','lend_volume':'shares','lend_sell_volume':'shares','lend_repay_volume':'shares'}
# Use the same reviewed declarations as the public adapter mapping.
from runpy import run_path
input_contracts=run_path(str(ROOT/'src/axiom_research/qsys_input_contracts.py'))['CORE_INPUT_CONTRACTS']
units.update({name: declaration['unit'] for name,declaration in input_contracts.items() if name in units})
deps={k:'market_daily.'+k for k in ('open','high','low','close','vwap','volume_shares','amount_cny')}
deps.update({k:'valuation_daily.'+k for k in ('pe','pb','ps','turnover_rate','total_mv','circ_mv')})
deps.update({k:'moneyflow_daily.'+k for k in ('net_inflow','big_inflow')})
deps.update({k:'margin_daily.'+k for k in units if k.startswith(('margin_','lend_'))})
deps['factor']='adjustment_factors.factor'
meta={k:(v,0,'base',{deps[k]}) for k,v in units.items()}
for n in cat['shared_nodes']:
 dep=set().union(*(meta[x][3] for x in n['inputs'])) if n['inputs'] else set()
 h=max((meta[x][1] for x in n['inputs']),default=0)
 if n['op']=='rolling':h+=n['params']['window']-int(n['params']['inclusive_current'])
 if n['op'] in ('shift','pct_change'):h+=n['params']['periods']
 meta[n['name']]=(n['column']['unit'],h,n['column']['stage'],dep)
cache={}; count=0

def node(op,args=(),params=None,unit=None,dtype='float64'):
 global count
 params=params or {};args=[globals().get('alias_inputs',{}).get(x,x) for x in args]
 u=unit or (meta[args[1 if op=='where' else 0]][0] if args else 'dimensionless')
 st='cross_sectional' if op.startswith('cs_') or any(meta[x][2]=='cross_sectional' for x in args) else 'base'
 key=json.dumps([op,args,params,u,dtype],sort_keys=True)
 if key in cache:return cache[key]
 count+=1;name=f'_qsys_{count:04d}';cache[key]=name
 h=max((meta[x][1] for x in args),default=0)
 if op.startswith('rolling'):h+=params['window']-int(params['inclusive_current'])
 if op in ('shift','pct_change'):h+=params['periods']
 meta[name]=(u,h,st,set().union(*(meta[x][3] for x in args)) if args else set())
 cat['shared_nodes'].append(dict(name=name,op=op,version='1',inputs=args,params=params,column=dict(name=name,dtype=dtype,unit=u,stage=st,missing='preserve')))
 return name

def const(v,u='dimensionless'):return node('constant',params={'value':v},unit=u)
def binary(op,a,b):
 if not isinstance(a,str):a=const(a,meta[b][0] if op in ('add','sub','gt','lt','eq') else 'dimensionless')
 if not isinstance(b,str):b=const(b,meta[a][0] if op in ('add','sub','gt','lt','eq') else 'dimensionless')
 ua,ub=meta[a][0],meta[b][0]
 u=ua
 if op=='divide':u='dimensionless' if ua==ub else ua if ub=='dimensionless' else f'{ua}/{ub}'
 if op=='mul':u=ub if ua=='dimensionless' else ua if ub=='dimensionless' else f'{ua}*{ub}'
 if op in ('gt','lt','eq','and','or'):u='dimensionless'
 return node(op,[a,b],{'zero':'missing'} if op=='divide' else {'missing':'preserve'} if op in ('gt','lt','eq','and','or') else {},u,'bool' if op in ('gt','lt','eq','and','or') else 'float64')
def add(a,b):return binary('add',a,b)
def sub(a,b):return binary('sub',a,b)
def div(a,b):return binary('divide',a,b)
def mul(a,b):return binary('mul',a,b)
def gt(a,b):return binary('gt',a,b)
def lt(a,b):return binary('lt',a,b)
def abs_(a):return node('abs',[a])
def fl(a):return node('to_float',[a],unit='dimensionless')
def shift(a,w):return node('shift',[a],{'periods':w})
def ret(a,w=1):return node('pct_change',[a],{'periods':w,'fill_method':'none','zero':'missing'},'dimensionless')
def roll(a,w,red='mean',minimum=None,ddof=None):return node('rolling',[a],dict(window=w,min_periods=minimum or w,inclusive_current=True,reduction=red,missing='skip' if minimum else 'propagate',ddof=(1 if red=='std' else 0) if ddof is None else ddof,ties='average'),'dimensionless' if red=='rank' else None)
def rank(a):return node('cs_rank',[a],dict(group='session',unknown_group='reject',missing='skip',ties='average',excluded='missing'),'dimensionless')
def maximum(a,b):
 if not isinstance(b,str):b=const(b,meta[a][0])
 return node('where',[gt(a,b),a,b],{'missing':'preserve'})
def minimum(a,b):
 if not isinstance(b,str):b=const(b,meta[a][0])
 return node('where',[lt(a,b),a,b],{'missing':'preserve'})
def clip(a,lo=None,hi=None):return node('clip',[a],dict(lower=lo,upper=hi))
def log(a):return node('log1p',[sub(div(a,const(1,meta[a][0])),1)],{'domain':'missing'},'dimensionless')
# Canonical daily source bindings: aliases are arithmetic recipes, never field substitution.
alias_inputs={'total_mv':'total_market_value_cny','circ_mv':'float_market_value_cny','net_inflow':'net_inflow_cny'}
for alias, canonical in alias_inputs.items():
 domain='money_flow_daily' if alias=='net_inflow' else 'valuation_daily'
 meta[canonical]=('CNY',0,'base',{domain+'.'+canonical})
 meta[alias]=meta[canonical]
 node_name=node('identity',[canonical],unit='CNY')
 # Existing expression lookup goes through this alias below.
 alias_inputs[alias]=node_name
for canonical in ('extra_large_buy_cny','extra_large_sell_cny'):
 meta[canonical]=('CNY',0,'base',{'money_flow_daily.'+canonical})
alias_inputs['big_inflow']=sub('extra_large_buy_cny','extra_large_sell_cny')
price={'open':'_a01_0004','high':'_a01_0006','low':'_a01_0008','close':'_a01_0010','amount':'_a01_0015','volume':'volume_shares','vwap':'vwap'}
for k in ('volume','vwap'):
 x=price[k]; condition=gt(x,0) if k=='vwap' else node('not',[lt(x,0)],{'missing':'preserve'},'dimensionless','bool')
 price[k]=node('where',[condition,x,const(None,meta[x][0])],{'missing':'preserve'})

def parse(expr):
 def visit(n):
  if isinstance(n,ast.Constant):return n.value
  if isinstance(n,ast.Name):return price.get(n.id,alias_inputs.get(n.id,n.id))
  if isinstance(n,ast.UnaryOp):return mul(visit(n.operand),-1)
  if isinstance(n,ast.BinOp):return {ast.Add:add,ast.Sub:sub,ast.Mult:mul,ast.Div:div}[type(n.op)](visit(n.left),visit(n.right))
  if isinstance(n,ast.Call):
   f=n.func.id;a=[visit(x) for x in n.args]
   if f=='Ref':return shift(*a)
   if f=='Abs':return abs_(*a)
   if f=='Log':return log(*a)
   if f=='Greater':return maximum(*a)
   if f=='Less':return minimum(*a)
   if f in ('Mean','Std','Sum','Max','Min','Rank'):return roll(a[0],a[1],f.lower() if f!='Mean' else 'mean')
   p=dict(window=a[-1],min_periods=a[-1],inclusive_current=True,missing='propagate')
   if f=='Corr':return node('rolling_corr',a[:2],{**p,'zero':'missing'},'dimensionless')
   p=dict(window=a[1],min_periods=a[1],inclusive_current=True,missing='propagate')
   if f in ('Slope','Rsquare','Resi'):return node('rolling_regression',a[:1],{**p,'output':f.lower(),'zero':'missing'},'dimensionless' if f=='Rsquare' else None)
   if f=='Quantile':return node('rolling_quantile',a[:1],{**p,'quantile':a[2],'interpolation':'linear'})
   if f in ('IdxMax','IdxMin'):return node('rolling_idx',a[:1],{**p,'extreme':f[3:].lower(),'ties':'first','index_base':1},'dimensionless')
  raise ValueError(ast.dump(n))
 return visit(ast.parse(expr.replace('$',''),mode='eval').body)

new={}; recipes={}; notes={}
def register(row,root,formula=None,note=None):
 fid=row['canonical_id']; group=fid[:3]
 if group not in groups:cat['groups'].append(dict(id=group,name=row['family'],numbering_policy='sparse_reserved_never_renumber'));groups.add(group)
 u,h,st,ds=meta[root]
 if any(d in ds for d in ('market_daily.open','market_daily.high','market_daily.low','market_daily.close','market_daily.vwap')):ds=ds|{'adjustment_factors.factor'}
 f=dict(group=group,id=fid,name=row['name'],formula=formula or row['expression'],dependencies=sorted(ds),lookback=h+1,normalization=copy.deepcopy(cat['features'][0]['normalization']),missing_policy='preserve missing; no implicit forward-fill or missing-to-zero; zero denominators missing; rolling policy explicit in DAG; frozen reference_members for CS',semantic_version='1.0.0',core_nodes=[dict(name=fid,op='identity',version='1',inputs=[root],params={},column=dict(name=fid,dtype='float64',unit=u,stage=st,missing='preserve'))])
 cat['features'].append(f);new[int(row['inventory_row'])]=f;recipes[row['name']]=root
 if note:notes[int(row['inventory_row'])]=note
existing={f['id']:f for f in cat['features']}
for r in rows[:158]:
 if r['canonical_id'] not in existing:register(r,parse(r['expression']),note='Alpha158 formula; Axiom complete rolling windows and preserve-missing policy, not Qlib partial-window equivalence.')
# Initial approved daily-domain passthroughs. Event/financial fields await report-period input adapter.
for r in rows[158:180]:
 key=r['name'].lstrip('$')
 if key in units and key!='lend_sell_volume':register(r,alias_inputs.get(key,key),note='Canonical CNY/shares units; pe remains supplier pe, not pe_ttm.')
c,o,h,l,v,a=[price[k] for k in ('close','open','high','low','volume','amount')]
r1=ret(c);up=fl(gt(r1,0));down=fl(lt(r1,0));rng=sub(h,l)
R={
'close_to_open_gap_1d':sub(div(o,shift(c,1)),1),'open_to_close_ret':sub(div(c,o),1),'close_pos_in_range':div(sub(c,l),rng),'open_pos_in_range':div(sub(o,l),rng),'upper_shadow_ratio':div(sub(h,maximum(o,c)),rng),'lower_shadow_ratio':div(sub(minimum(o,c),l),rng),'intraday_reversal_strength':mul(mul(r1,sub(div(c,o),1)),-1),
 'turnover_rate':'turnover_rate','amount_log':node('log1p',[div(clip(a,0),const(1,'CNY'))],{'domain':'missing'},'dimensionless'),
 'volume_shock_3':div(v,roll(v,3)), 'volume_shock_5':div(v,roll(v,5)), 'turnover_acceleration':sub('turnover_rate',shift('turnover_rate',3)), 'illiquidity':div(abs_(r1),a)}
for w in (1,3,5,20,60,120):R[f'ret_{w}d']=ret(c,w)
for w in (3,5):
 R[f'vol_mean_{w}d']=roll(v,w);R[f'amount_mean_{w}d']=roll(a,w)
for key in ('ret_1d','ret_3d','ret_5d','vol_mean_3d','vol_mean_5d','amount_mean_3d','amount_mean_5d'):R[key+'_rank']=rank(R[key])
for w in (20,60):R[f'volume_ratio_{w}d']=div(v,roll(v,w));R[f'amount_ratio_{w}d']=div(a,roll(a,w,minimum=10 if w==20 else 20))
for w in (120,250):R[f'distance_to_{w}d_high']=sub(div(c,roll(c,w,'max')),1)
for w in (60,120):
 m=w//3
 R[f'up_day_ratio_{w}d']=roll(up,w,minimum=m)
 R[f'trend_smoothness_{w}d']=div(ret(c,w),roll(abs_(r1),w,'sum',m))
 R[f'volatility_adjusted_return_{w}d']=div(ret(c,w),roll(r1,w,'std',m))
 R[f'rps_{w}d']=rank(ret(c,w))
R['rps_20d']=rank(ret(c,20));R['rps_20d_minus_rps_60d']=sub(R['rps_20d'],R['rps_60d'])
R['max_pullback_120d']=roll(sub(div(roll(c,120,'max',40),c),1),120,'max',40)
R['price_percentile_252d']=roll(c,252,'rank',60);R['distance_to_252d_low']=sub(div(c,roll(c,252,'min',60)),1)
not_up=fl(node('not',[gt(r1,0)],{'missing':'preserve'},'dimensionless','bool'))
R['volume_up_down_ratio_60d']=div(div(roll(mul(v,up),60,'sum',20),roll(up,60,'sum',20)),div(roll(mul(v,not_up),60,'sum',20),roll(not_up,60,'sum',20)))
R['above_avg_volume_ratio_60d']=roll(fl(gt(v,roll(v,60,minimum=20))),60,minimum=20)
R['volume_spike_20d']=div(sub(v,roll(v,20,minimum=10)),roll(v,20,'std',10))
R['volume_stability_60d']=div(roll(v,60,minimum=20),roll(v,60,'std',20))
R.update(log_mktcap=log('total_mv'),float_mktcap='circ_mv',pb_raw='pb')
for key in ('pe','pb'):
 R[key+'_rank_252d']=roll(key,252,'rank',60);R[key+'_delta_120d']=sub(key,shift(key,120));R[key+'_percentile_756d']=roll(key,756,'rank',180)
 R[key+'_distance_from_756d_low']=sub(div(key,roll(key,756,'min',180)),1)
 R[key+'_repair_room_to_median']=sub(div(roll(key,756,'median',180),key),1)
R['margin_balance_to_float_mv']=div('margin_balance','circ_mv')
for w in (20,60):R[f'margin_balance_chg_{w}d']=ret('margin_balance',w)
R['margin_buy_intensity_20d']=div(roll('margin_buy_amount',20,'sum',5),roll(a,20,'sum',5))
R['margin_repay_to_buy_20d']=div(roll('margin_repay_amount',20,'sum',5),roll('margin_buy_amount',20,'sum',5))
for w in (60,120):
 R[f'trend_consistency_{w}d']=roll(up,w,minimum=40)
 R[f'low_vol_uptrend_{w}d']=div(ret(c,w),roll(r1,w,'std',40,0))
 mx=roll(c,w,'max',40); dd=div(sub(mx,c),mx)
 R[f'return_drawdown_ratio_{w}d']=div(ret(c,w),roll(dd,w,'max',40))
 R[f'up_volume_down_volume_ratio_{w}d']=div(roll(mul(up,a),w,'sum',20),roll(mul(down,a),w,'sum',20))
mn=roll(c,60,'min',20);mx=roll(c,60,'max',20)
R['pullback_recovery_speed_60d']=div(sub(c,mn),sub(mx,mn))
R['new_high_persistence_120d']=roll(fl(gt(div(c,roll(c,120,'max',60)),.95)),120,minimum=60)
R['amount_stability_60d']=mul(div(roll(a,60,'std',20,0),roll(a,60,minimum=20)),-1)
z=div(sub(a,roll(a,120,minimum=20)),roll(a,120,'std',20,0))
R['breakout_volume_quality_120d']=mul(fl(gt(div(c,roll(c,120,'max',60)),.9)),fl(binary('and',gt(z,.5),lt(z,1.5))))
previous_high=shift(roll(c,252,'max',60),1)
R['breakout_252d_high']=fl(binary('or',gt(c,previous_high),binary('eq',c,previous_high)))
for r in rows[180:360]:
 if r['name'] in R:register(r,alias_inputs.get(R[r['name']],R[r['name']]),note='Reviewed source formula; explicit DAG windows, canonical CNY/shares inputs, and no pandas implicit fill or missing-to-false. CS uses frozen reference pool.')
for r in rows[360:]:
 expr=r['expression']
 leaves=set(json.loads(r['direct_dependencies']))
 if int(r['inventory_row']) in (380,388,387,384):continue
 if all(x in units or x in price for x in leaves):
  try:register(r,parse(expr),note='Raw QSYS expression in canonical CNY/shares units. This is not standard CS normalization or QSYS absnorm.')
  except (KeyError,ValueError) as e:print('uncompiled',r['name'],e)
# One shared financial bridge supplies two period-aligned typed operands.
# Report history is a bridge obligation, not a fabricated trading-session lag.
for leaf in ('ttm_revenue','ttm_revenue_prior_year'):
 meta[leaf]=('CNY',0,'base',{'financial_events.revenue'})
register(rows[183],sub(div('ttm_revenue','ttm_revenue_prior_year'),1),
 note='Fixed QSYS formula: TTM revenue / prior-year TTM revenue - 1; native revenue, not total_revenue; zero denominator and missing preserve. Data bridge uses four contiguous report quarters and the same cutoff-visible revisions for current and year-ago operands, strictly before session start. QSYS four-row shift and same-report revision filtering are not claimed fully equivalent; no trading-day lag. Source report history is a separate bridge obligation.')
for leaf in ('single_q_revenue','single_q_revenue_prior_year'):
 meta[leaf]=('CNY',0,'base',{'financial_events.revenue'})
register(rows[184],sub(div('single_q_revenue','single_q_revenue_prior_year'),1),
 note='Native revenue cumulative YTD converted by Data.single_quarter; prior-year operand is exactly four economic quarters earlier at the same visible cutoff. Missing components and zero denominators remain missing; no trading-day or report-row shift.')
# Definitions preserve all old records. Registry records certified aliases separately.
allids={f['id']:f for f in cat['features']}
registry=[]
for r in rows:
 i=int(r['inventory_row']); entry={k:r[k] for k in ('feature_key','name','family','registry_kind','expression','qsys_source','historical_pit')}
 entry.update(inventory_row=i,source_revision='852bcb73125f8781b7a80618b8188928ef4d38f2',leaf_dependencies=json.loads(r['leaf_dependencies']),field_bindings=json.loads(r['field_bindings']),candidate_id=r['canonical_id'],candidate_alias=r['candidate_alias_NOT_CERTIFIED'])
 if r['canonical_id'] in allids and i not in (384,387):entry.update(status='executable',selection={'id':r['canonical_id'],'semantic_version':allids[r['canonical_id']]['semantic_version']},semantic_review=notes.get(i,'Existing A01 immutable identity; complete windows and preserve-missing.'))
 elif i in (384,387):
  fid='LIQ001' if i==384 else 'ALP001';entry.update(status='alias',selection={'id':fid,'semantic_version':'1.0.0'},semantic_review='Certified exact passthrough/whitespace alias in migrated input semantics.')
 else:
  if i==179:code='supplier_quantity_unit';reason='rqmcl/lending sell native quantity has not been proven to be shares; no unit relabeling.'
  elif i in (274,276):code='ambiguous_valuation_name';reason='QSYS TTM-named feature reads non-TTM pe/ps. Supplier fields are distinct; no silent replacement or misleading name admitted.'
  elif i==388 or 'industry' in r['name'] or i in (204,205,249,250):code='historical_industry';reason='Historical taxonomy, membership intervals and publication cutoffs are not bound; current classifications cannot be backfilled.'
  elif i in (227,228,261,262,263,266):code='index_session_binding';reason='Index identity/calendar and one-row-per-session return calculation must be bound; QSYS repeated-row pct_change is not certified.'
  elif i in (259,260,264,265):code='reference_aggregation';reason='Reference-pool group aggregate/broadcast semantics require reviewed Core composition and frozen membership.'
  elif i in (341,342):code='source_formula_defect';reason='Source clips negative of a nonnegative series to nonnegative, yielding zero; meaningful corrected formula requires separately versioned decision.'
  elif i==380:code='adjustment_anchor_identity';reason='Raw factor requires explicit source/anchor interpretation; not a substitute for adjusted-price lineage.'
  elif i in range(206,213):code='limit_price_binding';reason='Historical upper/lower price limits with same price basis and cutoff are not yet bound.'
  elif i==189:code='last_true_age';reason='Requires reviewed last-true session-age semantics and missing/history policy.'
  elif i==188:code='breakout_missing_policy';reason='Source partial-window/exclusive-high boolean requires reviewed missing-to-false behavior before admission.'
  elif i==198 or i==199:code='rolling_policy_review';reason='Source rolling zscore helper policy still requires review.'
  elif any(x in r['name'] for x in ('holder','forecast')) or i in (312,325,326,327,330,345,346,347,348,349):code='event_revision_binding';reason='Requires report/event-period and previous-announcement revisions bound at each cutoff; no trading-row shift surrogate.'
  elif i in (308,309,310,311,318,319,320):code='dependent_composite';reason='Composite score depends on blocked inputs and/or source fill-zero/CS semantics requiring explicit review.'
  else:code='financial_report_binding';reason='Financial report-period/TTM/single-quarter fields and visible revisions are not bound to daily column input; year-over-year and acceleration cannot use 252-row shift.'
  entry.update(status='blocked',blocker={'code':code,'reason':reason},semantic_review='No executable placeholder is installed. Preserve source expression for future versioned migration.')
 registry.append(entry)
# prune orphan builder nodes (failed/blocked definitions are never recipes)
by={n['name']:n for n in cat['shared_nodes']}; needed=set()
def include(n):
 if n in by and n not in needed:
  needed.add(n)
  for p in by[n]['inputs']:include(p)
for f in cat['features']:
 for n in f['core_nodes']:
  for p in n['inputs']:include(p)
cat['shared_nodes']=[n for n in cat['shared_nodes'] if n['name'] in needed]
for name,obj in [('qsys_stock_v3.json',cat),('qsys_registry_v3.json',dict(schema_version='qsys_registry_v1',source_revision='852bcb73125f8781b7a80618b8188928ef4d38f2',normalization_boundary='normalized=True means Axiom frozen-reference CS zscore. QSYS absnorm expressions remain separate raw recipes; never aliases for normalized=True.',rows=registry))]:
 (ROOT/'src/axiom_research/catalogs'/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
print('features',len(cat['features']),'registry',len(registry),'states',{s:sum(r['status']==s for r in registry)for s in ('executable','alias','blocked')})
