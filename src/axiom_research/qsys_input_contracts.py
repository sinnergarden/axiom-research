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
