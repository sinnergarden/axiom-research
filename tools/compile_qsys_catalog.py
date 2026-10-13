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
cat['catalog_id']='qsys_stock_v4'
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
# Shared report bridge: native reported values and explicit economic-period lags.
contracts=run_path(str(ROOT/'src/axiom_research/qsys_input_contracts.py'))['FINANCIAL_FIELD_CONTRACTS']
for contract in contracts.values():
 for leaf in contract['outputs']:
  meta[leaf]=(contract['unit'],0,'base',{contract['domain']+'.'+contract['field']})
financial_aliases={k:'reported_'+v for k,v in {
 'roe':'roe','grossprofit_margin':'grossprofit_margin','debt_to_assets':'debt_to_assets',
 'current_ratio':'current_ratio','net_income':'net_income','revenue':'revenue',
 'total_assets':'total_assets','equity':'parent_equity','op_cashflow':'operating_cash_flow'}.items()}
alias_inputs.update(financial_aliases)
financial={}
for name,leaf in financial_aliases.items():financial['$'+name]=leaf
financial.update(roe='reported_roe',gross_margin='reported_grossprofit_margin',debt_to_asset='reported_debt_to_assets',
 net_margin=div('reported_net_income','reported_revenue'),
 operating_cf_to_profit=div('reported_operating_cash_flow','reported_net_income'),
 roa=div('reported_net_income','reported_total_assets'),
 earnings_yield_proxy=div('reported_net_income','total_mv'),
 ocf_margin=div('reported_operating_cash_flow','reported_revenue'))
for native,name in (('revenue','revenue_yoy'),('net_income','profit_yoy'),('inventory','inventory_yoy'),('accounts_receiv','ar_yoy')):
 financial[name]=sub(div('reported_'+native,'reported_'+native+'_prior_year'),1)
financial['peg_proxy']=div('pe',abs_(financial['profit_yoy']))
financial['gross_margin_delta_yoy']=sub(
 div(sub('single_q_revenue','single_q_oper_cost'),'single_q_revenue'),
 div(sub('single_q_revenue_prior_year','single_q_oper_cost_prior_year'),'single_q_revenue_prior_year'))
for row in rows:
 if row['name'] in financial:
  register(row,financial[row['name']],note='Data-selected native report values retain units and basis. Current and prior-year operands use exact report-quarter offsets and cutoff-visible revisions, not trading-row shifts. Zero denominators/missing remain missing; no implicit TTM conversion. Separate statement fields retain their own visible report lineage.')
  if row['name'] in ('revenue_yoy','profit_yoy','inventory_yoy','ar_yoy','peg_proxy'):
   new[int(row['inventory_row'])]['semantic_version']='2.0.0'
for row in rows[367:375]:
 register(row,parse(row['expression']),note='Original absnorm arithmetic on native reported financial inputs in CNY. This does not convert cumulative YTD values into TTM; each selected report/revision remains in input evidence.')
# Source-explicit missing fills apply only inside these composite scores.
def filled(a,value=0):return node('fill',[a],{'value':value})
def csz(a):return node('cs_zscore',[a],dict(group='session',unknown_group='reject',missing='fill_zero',ddof=0,epsilon=1e-12,constant='zero',clip=None,excluded='missing'),'dimensionless')
R['continuation_candidate_score']=div(add(add(rank(clip(filled(R['trend_smoothness_60d']),-1,1)),filled(R['rps_120d'])),add(mul(filled(R['price_percentile_252d']),.5),mul(sub(1,filled(R['rps_20d'])),.3))),2.8)
R['repair_candidate_score']=div(add(add(sub(1,filled(R['pe_percentile_756d'],.5)),mul(sub(1,clip(filled(R['distance_to_252d_low']),0,1)),.5)),mul(filled(R['rps_120d']),.3)),1.8)
R['overheat_risk_score']=add(add(mul(filled(R['price_percentile_252d']),.4),mul(filled(R['rps_120d']),.4)),mul(div(clip(filled(R['volume_spike_20d']),0,3),3),.2))
R['margin_crowding_score']=add(csz(R['margin_balance_to_float_mv']),csz(R['margin_balance_chg_60d']))
R['margin_trend_confirm_score']=mul(csz(R['margin_balance_chg_60d']),clip(csz(R['ret_60d']),0))
R['margin_overheat_risk_score']=mul(csz(R['margin_crowding_score']),clip(csz(R['ret_120d']),0))
R['margin_pullback_recovery_confirm']=mul(R['margin_trend_confirm_score'],clip(csz(R['pullback_recovery_speed_60d']),0))
for row in rows:
 if row['name'] in ('continuation_candidate_score','repair_candidate_score','overheat_risk_score','margin_crowding_score','margin_trend_confirm_score','margin_overheat_risk_score','margin_pullback_recovery_confirm'):
  register(row,R[row['name']],note='Source-explicit intermediate fill-zero (valuation percentile fill 0.5); CS uses frozen reference membership, population std and constant zero. Raw operands retain missing evidence; score values do not certify input completeness.')
# Explicitly day-named deltas retain the source trading-session semantics;
# these are differences of historical PIT observations, not fiscal YoY.
for src,name,lag in (('roe','roe_delta_252d',252),('grossprofit_margin','grossprofit_margin_delta_252d',252),('debt_to_assets','debt_to_assets_delta_252d',252),('op_cashflow','op_cashflow_delta_252d',252),('roe','roe_delta_756d',756),('net_margin','net_margin_delta_756d',756)):
 root=financial_aliases[src] if src in financial_aliases else financial[src]
 R[name]=sub(root,shift(root,lag))
for src,name in (('revenue','revenue_yoy_accel'),('net_income','profit_yoy_accel')):
 current=sub(div('reported_'+src,'reported_'+src+'_prior_year'),1)
 previous=sub(div('reported_'+src+'_prior_year','reported_'+src+'_two_years_ago'),1)
 R[name]=sub(current,previous)
R['value_trap_risk_score']=add(add(mul(sub(1,filled(R['pe_percentile_756d'],.5)),.3),mul(sub(1,filled(R['price_percentile_252d'],.5)),.3)),add(mul(fl(lt(filled(R['roe_delta_756d']),-.01)),.2),mul(fl(lt(filled(R['net_margin_delta_756d']),-.01)),.2)))
for row in rows:
 if row['name'] in ('roe_delta_252d','grossprofit_margin_delta_252d','debt_to_assets_delta_252d','op_cashflow_delta_252d','roe_delta_756d','net_margin_delta_756d','revenue_yoy_accel','profit_yoy_accel','value_trap_risk_score'):
  register(row,R[row['name']],note='Day-named deltas compare cutoff-visible daily financial observations exactly 252/756 sessions apart, not fiscal YoY. YoY acceleration instead compares economic quarters 0/4/8 at one visible cutoff; semantic v2 replaces the source daily-row approximation. Value-trap score preserves source-explicit intermediate fills and thresholds in native units.')
  if row['name'] in ('revenue_yoy_accel','profit_yoy_accel'):
   new[int(row['inventory_row'])]['semantic_version']='2.0.0'
# Historical exchange limits and original prices share an unadjusted basis.
for leaf in ('up_limit','down_limit'):meta[leaf]=('CNY/share',0,'base',{'price_limits.'+leaf})
for leaf in ('raw_open','raw_close'):meta[leaf]=('CNY/share',0,'base',{'market_daily.'+leaf[4:]})
R['is_limit_up']=binary('and',gt('up_limit',0),binary('or',gt('raw_close','up_limit'),binary('eq','raw_close','up_limit')))
R['is_limit_down']=binary('and',gt('down_limit',0),binary('or',lt('raw_close','down_limit'),binary('eq','raw_close','down_limit')))
R['distance_to_limit_up']=div(sub('up_limit','raw_close'),'raw_close')
R['distance_to_limit_down']=div(sub('raw_close','down_limit'),'raw_close')
R['limit_up_count_5d']=roll(fl(R['is_limit_up']),5,'sum')
R['opened_from_limit_up']=binary('and',R['is_limit_up'],lt('raw_open','up_limit'))
for row in rows:
 if row['name'] in ('is_limit_up','is_limit_down','distance_to_limit_up','distance_to_limit_down','limit_up_count_5d','opened_from_limit_up'):
  root=R[row['name']]
  if meta[root][0]=='dimensionless' and row['name'] in ('is_limit_up','is_limit_down','opened_from_limit_up'):root=fl(root)
  register(row,root,note='Original unadjusted native open/close versus historical exchange limits in CNY/share, selected at declared daily cutoff. Missing price/limit propagates rather than becoming false; semantic v2 corrects source boolean missing conversion. Five-day counts require all five observations.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
  new[int(row['inventory_row'])]['dependencies']=[d for d in new[int(row['inventory_row'])]['dependencies'] if d!='adjustment_factors.factor']
# A single shared Core aggregate broadcasts only to frozen reference members.
def csmean(root):return node('cs_mean',[root],dict(group='session',unknown_group='reject',missing='skip',excluded='missing'))
def masked(root,condition):return node('where',[condition,root,div(const(0,meta[root][0]),0)],{'missing':'preserve'})
R['market_breadth']=csmean(fl(gt(ret(c),0)))
R['limit_up_breadth']=csmean(fl(R['is_limit_up']))
for field,name,reverse in (('circ_mv','small_vs_large_strength',False),('pb','growth_vs_value_proxy',True)):
 ranked=rank(alias_inputs.get(field,field))
 low=binary('or',lt(ranked,.3),binary('eq',ranked,.3))
 high=binary('or',gt(ranked,.7),binary('eq',ranked,.7))
 low_mean=csmean(masked(ret(c),low));high_mean=csmean(masked(ret(c),high))
 R[name]=sub(high_mean,low_mean) if reverse else sub(low_mean,high_mean)
for row in rows:
 if row['name'] in ('market_breadth','limit_up_breadth','small_vs_large_strength','growth_vs_value_proxy'):
  register(row,R[row['name']],note='Session mean over frozen reference members only; broadcast back to reference members, excluded rows stay missing. Missing observations are skipped, empty cohorts remain missing. Rank thresholds retain average ties. v2 repairs source group-result alignment and missing-return-to-false behavior.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
# Scores explicitly preserve source fills only at named intermediate nodes.
for row in rows:
 if row['name'] in ('continuation_candidate_score','repair_candidate_score','overheat_risk_score','margin_crowding_score','margin_trend_confirm_score','margin_overheat_risk_score','margin_pullback_recovery_confirm','value_trap_risk_score'):
  new[int(row['inventory_row'])]['missing_policy']='raw inputs preserve missing; source-explicit intermediate fill and constant-group policies are encoded in DAG; score validity is not input-completeness evidence'
# The benchmark bridge binds one explicit index and one observation per session.
meta['benchmark_close']=('index points',0,'base',{'benchmark_daily.close'})
for w in (3,5):R[f'stock_minus_index_ret_{w}d']=sub(ret(c,w),ret('benchmark_close',w))
for w in (5,10,20):R[f'index_volatility_{w}']=roll(ret('benchmark_close'),w,'std')
R['market_trend_strength']=ret('benchmark_close',20)
for row in rows:
 if row['name'] in ('stock_minus_index_ret_3d','stock_minus_index_ret_5d','index_volatility_5','index_volatility_10','index_volatility_20','market_trend_strength'):
  register(row,R[row['name']],note='Explicit caller-bound benchmark identity is part of input semantics; one native unadjusted index close per stock-calendar session is broadcast. Returns/volatility operate along sessions, never repeated stock rows. Missing sessions remain missing; v2 corrects source repeated-row/index alignment.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
# Recovered tracked helper: SysQ transforms blob bc225492ee6d9d63718ca65a95c941d3de2ef77c.
R['amount_zscore_20']=div(sub(a,roll(a,20)),roll(a,20,'std',ddof=0))
R['is_profitable_ttm']=fl(gt('ttm_net_income',0))
for row in rows:
 if row['name'] in ('amount_zscore_20','is_profitable_ttm'):
  register(row,R[row['name']],note='Amount zscore uses complete 20-session windows and population std; zero std stays missing, matching recovered tracked SysQ helper blob bc225492ee6d9d63718ca65a95c941d3de2ef77c (not asserted present in pinned export). Profitability uses Data-derived native net-income TTM and preserves missing instead of false; v2 explicitly repairs source NaN comparison.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
# Visible report observations are sequenced by distinct report period; revisions
# never masquerade as an extra previous-quarter observation (explicit v2).
R['holder_num_chg_qoq']=sub(div('holder_count','holder_count_previous1'),1)
R['holder_num_chg_2q']=sub(div('holder_count','holder_count_previous2'),1)
R['avg_shares_per_holder_chg_qoq']=sub(div('holder_count_previous1','holder_count'),1)
R['top10_holder_ratio_chg_qoq']=sub(div('top10_ratio','top10_ratio_previous1'),1)
R['top10_holder_ratio']='top10_ratio'
R['holder_concentration_score']=div(add(add(mul(csz(R['holder_num_chg_qoq']),-1),csz(R['avg_shares_per_holder_chg_qoq'])),csz(R['top10_holder_ratio_chg_qoq'])),3)
R['holder_squeeze_score']=mul(mul(csz(R['holder_num_chg_qoq']),-1),clip(csz(R['ret_60d']),0))
R['holder_price_confirm_score']=mul(csz(R['holder_concentration_score']),clip(csz(R['ret_120d']),0))
R['holder_concentration_trend_confirm']=mul(R['holder_concentration_score'],clip(csz(R['trend_consistency_120d']),0))
R['holder_concentration_low_vol_uptrend']=mul(R['holder_concentration_score'],clip(csz(R['low_vol_uptrend_120d']),0))
R['margin_holder_trend_confirm']=mul(R['margin_trend_confirm_score'],clip(csz(R['holder_concentration_score']),0))
for row in rows:
 if row['name'] in ('holder_num_chg_qoq','holder_num_chg_2q','avg_shares_per_holder_chg_qoq','top10_holder_ratio_chg_qoq','top10_holder_ratio','holder_concentration_score','holder_squeeze_score','holder_price_confirm_score','holder_concentration_trend_confirm','holder_concentration_low_vol_uptrend','margin_holder_trend_confirm'):
  register(row,R[row['name']],note='v2 uses latest visible distinct report periods, with current cutoff-selected revisions, not trading shifts or corrections counted as new reports. Report lags mean previous observed report, not a guaranteed contiguous calendar quarter. Data preserves top10 completeness. Source composite fills are explicit; all three concentration components are required in the declared recipe.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
  if row['name'] not in ('holder_num_chg_qoq','holder_num_chg_2q','avg_shares_per_holder_chg_qoq','top10_holder_ratio_chg_qoq','top10_holder_ratio'):
   new[int(row['inventory_row'])]['missing_policy']='raw report operands preserve missing; source-explicit intermediate CS fill-zero and constant-group-zero; score validity is not complete report coverage'
meta['session_ordinal']=('days',0,'base',set())
for prefix,name in (('holder','holder_num_stale_days'),('top10','top10_holder_stale_days')):
 for row in rows:
  if row['name']==name:
   register(row,clip(sub('session_ordinal',prefix+'_announcement_ordinal'),0),note='Calendar-day age of the selected visible latest distinct report announcement; native date encoded as Gregorian ordinal, subtraction remains in Core. Missing announcement remains missing; report corrections retain their selected announcement date.')
   new[int(row['inventory_row'])]['semantic_version']='2.0.0'
# Explicit v2 corrections: true supplier TTM fields are not aliases of pe/ps.
for leaf in ('pe_ttm','ps_ttm'):meta[leaf]=('ratio',0,'base',{'valuation_daily.'+leaf})
meta['raw_factor']=('dimensionless',0,'base',{'adjustment_factors.factor'})
for row in rows:
 if row['name'] in ('pe_ttm','ps_ttm','$factor'):
  register(row,'raw_factor' if row['name']=='$factor' else row['name'],note='Explicit v2 migration: TTM-named valuation features use their distinct native supplier TTM fields, not pe/ps aliases. Raw factor exposes the native supplier adjustment multiplier with its profile lineage, not an adjusted price or output-day normalized factor.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
meta['is_suspended']=('dimensionless',0,'base',{'security_status.is_suspended'})
for row in rows:
 if row['name']=='tradability_score':
  register(row,clip(sub(sub(sub(1,'is_suspended'),mul(fl(R['is_limit_up']),.5)),mul(fl(R['is_limit_down']),.5)),0,1),note='v2 uses native historical suspension state and native-price limit flags. Boolean suspension is exactly 0/1; absent status remains missing, not assumed trading. Preserve unknown limits/status instead of source implicit fill-zero.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
score=const(0)
for code,value in ((1,2),(2,1),(3,1.5),(4,.5),(5,-1),(6,-1),(7,-2),(8,-2)):
 score=node('where',[binary('eq','forecast_category_code',code),const(value),score],{'missing':'preserve'})
forecast_roots={'forecast_type_score':score,'forecast_stale_days':clip(sub('session_ordinal','forecast_announcement_ordinal'),0),'has_forecast':fl(gt('forecast_announcement_ordinal',0))}
for row in rows:
 if row['name'] in forecast_roots:
  register(row,forecast_roots[row['name']],note='v2 selects latest cutoff-visible announcement, ties by report period, with Data revision selection; future forecast report periods are allowed. Native categories use stable codes, scoring remains Core (2,1,1.5,0.5,-1,-1,-2,-2; unknown supplied category 0). Absent/unknown announcement remains missing, not fabricated 0/999. Presence is 1 only for a visible supplied announcement.')
  new[int(row['inventory_row'])]['semantic_version']='2.0.0'
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
  elif i==186:code='profitability_missing_policy';reason='Source turns absent profitability inputs into false/zero; definition requires an explicit missing-input policy before admission.'
  elif i in (259,260,264,265):code='reference_aggregation';reason='Reference-pool group aggregate/broadcast semantics require reviewed Core composition and frozen membership.'
  elif i in (341,342):code='source_formula_defect';reason='Source clips negative of a nonnegative series to nonnegative, yielding zero; meaningful corrected formula requires separately versioned decision.'
  elif i==380:code='adjustment_anchor_identity';reason='Raw factor requires explicit source/anchor interpretation; not a substitute for adjusted-price lineage.'
  elif i in range(206,213):code='limit_price_binding';reason='Historical upper/lower price limits with same price basis and cutoff are not yet bound.'
  elif i==189:code='last_true_age';reason='Requires reviewed last-true session-age semantics and missing/history policy.'
  elif i==188:code='breakout_missing_policy';reason='Source partial-window/exclusive-high boolean requires reviewed missing-to-false behavior before admission.'
  elif i==198 or i==199:code='rolling_policy_review';reason='Fixed source inventory imports rolling_zscore, but its helper implementation is absent from the pinned source export; min_periods, ddof and zero-scale policy are not established.'
  elif i==312:code='eligibility_semantics';reason='Observed margin balance presence is not authoritative margin eligibility; no qualification flag is fabricated.'
  elif i==347:code='blocked_formula_dependency';reason='Depends on volume_contraction_after_rise_60d, whose original formula is provably constant zero and needs an explicit corrected definition.'
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
for name,obj in [('qsys_stock_v4.json',cat),('qsys_registry_v4.json',dict(schema_version='qsys_registry_v1',source_revision='852bcb73125f8781b7a80618b8188928ef4d38f2',normalization_boundary='normalized=True means Axiom frozen-reference CS zscore. QSYS absnorm expressions remain separate raw recipes; never aliases for normalized=True.',rows=registry))]:
 (ROOT/'src/axiom_research/catalogs'/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
print('features',len(cat['features']),'registry',len(registry),'states',{s:sum(r['status']==s for r in registry)for s in ('executable','alias','blocked')})
