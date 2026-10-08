"""Synthetic values in the captured canonical owner's saved field shape."""
from copy import deepcopy
import json
from pathlib import Path

SHAPE = json.loads((Path(__file__).parent/'fixtures'/'stock_compact_owner_wire_shape.json').read_text())


def saved_context(*, snapshot, pit, universe, sessions, anchor, cutoff):
    native = dict(adjustment_anchor=None, cutoff_by_session=dict.fromkeys(sessions,cutoff),
        fields=['open','close'], pit_policy=pit, policy_by_session=None,
        price_basis='unadjusted', purpose='label_outcomes', sessions=list(sessions),
        symbols=list(universe), universe_id=None)
    factor = {**deepcopy(native), 'fields':['factor']}
    query = {**deepcopy(native), 'price_basis':'common_anchor_adjusted_v1','adjustment_anchor':anchor}
    derivation = dict(anchor_session=anchor, decision_cutoff=cutoff, decision_session=sessions[-1],
        factor_contract_id='synthetic.adjustment_factors.v1', factor_domain='adjustment_factors',
        factor_field='factor', factor_query=factor, factor_reader_version='synthetic_native/1',
        factor_source_profile_id='synthetic.factor.v1', formula='price_t * factor_t / factor_anchor',
        price_query=native, recipe_version='common_anchor_price_v1')
    context = dict(contract_id='synthetic.market_daily.v1', derivation=derivation, domain='market_daily',
        limitations=['Synthetic values; engineering evidence only.'], query=query,
        reader_version='synthetic_native/1', snapshot_id=snapshot, source_profile_id='synthetic.price.v1')
    assert sorted(context) == SHAPE['context_fields']
    assert sorted(query) == SHAPE['query_fields']
    assert sorted(native) == sorted(factor) == SHAPE['native_query_fields']
    assert sorted(derivation) == SHAPE['derivation_fields']
    return context


def native_data_fixture(base):
    """Keep the upstream DataBatch ABI; let the real owner project saved context."""
    class NativeDataFixture(base):
        def read(self, *, snapshot, query):
            result = super().read(snapshot=snapshot,query=query)
            result.context['query'] = {k:deepcopy(result.context['query'][k]) for k in SHAPE['native_query_fields']}
            assert 'domain' not in result.context['query']
            return result

        def adjust(self, price, factors, **kwargs):
            result = super().adjust(price,factors,**kwargs)
            result.context['derivation'].update(
                factor_domain=factors.context['domain'], factor_contract_id=factors.context['contract_id'],
                factor_reader_version=factors.context['reader_version'],
                factor_source_profile_id=factors.context['source_profile_id'])
            # contract_version belongs to the incoming DataBatch. The real
            # owner's _source_view deliberately excludes it from saved context.
            projected = {k:result.context[k] for k in SHAPE['context_fields']}
            assert sorted(projected['derivation']) == SHAPE['derivation_fields']
            assert all(sorted(projected['derivation'][name]) == SHAPE['native_query_fields']
                       for name in ('price_query','factor_query'))
            return result
    return NativeDataFixture
