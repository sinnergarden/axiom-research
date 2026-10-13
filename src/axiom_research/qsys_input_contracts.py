"""Explicit canonical input contracts shared by QSYS compilation and adapters.

No default unit inference is allowed: new fields require a reviewed declaration.
Data physical dtypes are distinct from Core's float64 numerical fact schema.
"""

DAILY_FIELD_CONTRACTS = {
    'volume_shares': {'domain': 'market_daily', 'dtype': 'int64', 'unit': 'shares'},
    **{field: {'domain': 'valuation_daily', 'dtype': 'float64', 'unit': 'ratio'}
       for field in ('pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'volume_ratio')},
    **{field: {'domain': 'valuation_daily', 'dtype': 'float64', 'unit': 'percent'}
       for field in ('turnover_rate', 'turnover_rate_f')},
    **{field: {'domain': 'valuation_daily', 'dtype': 'float64', 'unit': 'CNY'}
       for field in ('total_market_value_cny', 'float_market_value_cny')},
    **{field: {'domain': 'valuation_daily', 'dtype': 'float64', 'unit': 'shares'}
       for field in ('total_shares', 'float_shares', 'free_shares')},
    **{field: {'domain': 'money_flow_daily', 'dtype': 'float64', 'unit': 'CNY'}
       for field in ('net_inflow_cny', 'extra_large_buy_cny', 'extra_large_sell_cny')},
    **{field: {'domain': 'margin_daily', 'dtype': 'float64', 'unit': 'CNY'}
       for field in ('margin_balance', 'margin_buy_amount', 'margin_repay_amount', 'margin_total_balance')},
    **{field: {'domain': 'margin_daily', 'dtype': 'float64', 'unit': 'shares'}
       for field in ('lend_volume', 'lend_repay_volume')},
}
CORE_INPUT_CONTRACTS = {
    **{field: {'domain': 'market_daily', 'dtype': 'float64', 'unit': 'CNY/share'}
       for field in ('open', 'high', 'low', 'close', 'vwap')},
    'amount_cny': {'domain': 'market_daily', 'dtype': 'float64', 'unit': 'CNY'},
    'factor': {'domain': 'adjustment_factors', 'dtype': 'float64', 'unit': 'dimensionless'},
    **{field: {**spec, 'dtype': 'float64'} for field, spec in DAILY_FIELD_CONTRACTS.items()},
}

# Shared by catalog selection and the generic financial materialization bridge.
# Query scope, symbols and cutoff come from the caller's frozen source scope.
FINANCIAL_FIELD_CONTRACTS = {
    'revenue_ttm_yoy': {
        'domain': 'financial_events', 'field': 'revenue', 'endpoint': 'income',
        'report_type': '1', 'basis': 'cumulative_ytd', 'unit': 'CNY',
        'session_timezone': 'Asia/Shanghai',
        'visibility': 'strict_before_session_start',
        'outputs': {
            'ttm_revenue': {'derivation': 'ttm', 'quarters_back': 0},
            'ttm_revenue_prior_year': {'derivation': 'ttm', 'quarters_back': 4},
            'single_q_revenue': {'derivation': 'single_quarter', 'quarters_back': 0},
            'single_q_revenue_prior_year': {'derivation': 'single_quarter', 'quarters_back': 4},
        },
    },
}
CORE_INPUT_CONTRACTS.update({
    field: {'domain': 'financial_events', 'dtype': 'float64', 'unit': 'CNY'}
    for field in ('revenue', 'ttm_revenue', 'ttm_revenue_prior_year', 'single_q_revenue', 'single_q_revenue_prior_year')
})

# Reported inputs retain native basis; no implicit TTM or percent conversion.
_REPORT_INPUTS = (
    ('revenue', 'financial_events', 'income', 'CNY', 'cumulative_ytd', True),
    ('net_income', 'financial_events', 'income', 'CNY', 'cumulative_ytd', True),
    ('total_assets', 'balance_sheet_events', 'balancesheet', 'CNY', 'period_end_stock', False),
    ('parent_equity', 'balance_sheet_events', 'balancesheet', 'CNY', 'period_end_stock', False),
    ('inventory', 'balance_sheet_events', 'balancesheet', 'CNY', 'period_end_stock', True),
    ('accounts_receiv', 'balance_sheet_events', 'balancesheet', 'CNY', 'period_end_stock', True),
    ('operating_cash_flow', 'cash_flow_events', 'cashflow', 'CNY', 'cumulative_ytd', False),
    ('roe', 'financial_indicator_events', 'fina_indicator', 'percent', 'reported', False),
    ('grossprofit_margin', 'financial_indicator_events', 'fina_indicator', 'percent', 'reported', False),
    ('debt_to_assets', 'financial_indicator_events', 'fina_indicator', 'percent', 'reported', False),
    ('current_ratio', 'financial_indicator_events', 'fina_indicator', 'ratio', 'reported', False),
)
for _field, _domain, _endpoint, _unit, _basis, _year_ago in _REPORT_INPUTS:
    _outputs={'reported_'+_field: {'derivation':'reported','quarters_back':0}}
    if _year_ago:
        _outputs['reported_'+_field+'_prior_year']={'derivation':'reported','quarters_back':4}
    if _field=='revenue':
        FINANCIAL_FIELD_CONTRACTS['revenue_ttm_yoy']['outputs'].update(_outputs)
    else:
        FINANCIAL_FIELD_CONTRACTS['reported_'+_field]={
            'domain':_domain,'field':_field,'endpoint':_endpoint,
            'report_type':None if _endpoint=='fina_indicator' else '1',
            'basis':_basis,'unit':_unit,'session_timezone':'Asia/Shanghai',
            'visibility':'strict_before_session_start','outputs':_outputs}
    CORE_INPUT_CONTRACTS.update({leaf:{'domain':_domain,'dtype':'float64','unit':_unit} for leaf in (_field,*_outputs)})
FINANCIAL_FIELD_CONTRACTS['single_q_oper_cost']={
    'domain':'financial_events','field':'oper_cost','endpoint':'income','report_type':'1',
    'basis':'cumulative_ytd','unit':'CNY','session_timezone':'Asia/Shanghai',
    'visibility':'strict_before_session_start',
    'outputs':{'single_q_oper_cost':{'derivation':'single_quarter','quarters_back':0},
               'single_q_oper_cost_prior_year':{'derivation':'single_quarter','quarters_back':4}}}
CORE_INPUT_CONTRACTS.update({leaf:{'domain':'financial_events','dtype':'float64','unit':'CNY'}
    for leaf in ('oper_cost',*FINANCIAL_FIELD_CONTRACTS['single_q_oper_cost']['outputs'])})

# Acceleration compares two fiscal YoY observations at one visible cutoff.
for _key in ('revenue_ttm_yoy','reported_net_income'):
    _contract=FINANCIAL_FIELD_CONTRACTS[_key]
    _leaf='reported_'+_contract['field']+'_two_years_ago'
    _contract['outputs'][_leaf]={'derivation':'reported','quarters_back':8}
    CORE_INPUT_CONTRACTS[_leaf]={'domain':_contract['domain'],'dtype':'float64','unit':_contract['unit']}

DAILY_FIELD_CONTRACTS.update({field:{'domain':'price_limits','dtype':'float64','unit':'CNY/share'}
    for field in ('up_limit','down_limit')})
CORE_INPUT_CONTRACTS.update({field:dict(spec) for field,spec in DAILY_FIELD_CONTRACTS.items() if field in ('up_limit','down_limit')})
CORE_INPUT_CONTRACTS.update({field:{'domain':'market_daily','dtype':'float64','unit':'CNY/share'} for field in ('raw_close','raw_open')})

CORE_INPUT_CONTRACTS['benchmark_close']={'domain':'benchmark_daily','dtype':'float64','unit':'index points'}

FINANCIAL_FIELD_CONTRACTS['reported_net_income']['outputs']['ttm_net_income']={'derivation':'ttm','quarters_back':0}
CORE_INPUT_CONTRACTS['ttm_net_income']={'domain':'financial_events','dtype':'float64','unit':'CNY'}

# Distinct report observations: corrections remain the same report, never a new lag.
for _field,_domain,_unit,_prefix,_lags in (
    ('holder_num','shareholder_count_events','holders','holder_count',3),
    ('top10_ratio','top_holders_reports','percent of total shares','top10_ratio',2)):
    _outputs={_prefix+('' if i==0 else '_previous'+str(i)):{'derivation':'report_observation','reports_back':i} for i in range(_lags)}
    FINANCIAL_FIELD_CONTRACTS[_prefix]={'domain':_domain,'field':_field,'endpoint':None,
        'report_type':None,'basis':'reported','unit':_unit,'session_timezone':'Asia/Shanghai',
        'visibility':'strict_before_session_start','outputs':_outputs}
    CORE_INPUT_CONTRACTS.update({leaf:{'domain':_domain,'dtype':'float64','unit':_unit} for leaf in (_field,*_outputs)})

for _domain,_prefix in (('shareholder_count_events','holder'),('top_holders_reports','top10')):
    _leaf=_prefix+'_announcement_ordinal'
    FINANCIAL_FIELD_CONTRACTS[_leaf]={'domain':_domain,'field':'announcement_date','endpoint':None,
        'report_type':None,'basis':'reported','unit':'days','session_timezone':'Asia/Shanghai',
        'visibility':'strict_before_session_start','outputs':{_leaf:{'derivation':'report_announcement_ordinal','reports_back':0}}}
    CORE_INPUT_CONTRACTS[_leaf]={'domain':_domain,'dtype':'float64','unit':'days'}
CORE_INPUT_CONTRACTS['session_ordinal']={'domain':'calendar','dtype':'float64','unit':'days'}

CORE_INPUT_CONTRACTS['raw_factor']={'domain':'adjustment_factors','dtype':'float64','unit':'dimensionless'}

DAILY_FIELD_CONTRACTS['is_suspended']={'domain':'security_status','dtype':'bool','unit':None}
CORE_INPUT_CONTRACTS['is_suspended']={'domain':'security_status','dtype':'float64','unit':'dimensionless'}

for _field,_leaf,_unit,_derivation in (
    ('forecast_type','forecast_category_code','dimensionless','forecast_category_code'),
    ('announcement_date','forecast_announcement_ordinal','days','report_announcement_ordinal')):
    FINANCIAL_FIELD_CONTRACTS[_leaf]={'domain':'forecast_events','field':_field,'endpoint':None,
        'report_type':None,'basis':'reported','unit':_unit,'session_timezone':'Asia/Shanghai',
        'visibility':'strict_before_session_start','outputs':{_leaf:{'derivation':_derivation,'reports_back':0}}}
    CORE_INPUT_CONTRACTS[_leaf]={'domain':'forecast_events','dtype':'float64','unit':_unit}
