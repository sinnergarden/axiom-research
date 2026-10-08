"""Resolved stock Label definitions; no Data, numerical or model execution."""
from copy import deepcopy

from .contracts import LabelSpec, MaturitySpec
from .api import from_dict, semantic_identity, to_dict, validate
from .stock_fold_inputs import require
from .stock_label_contracts import NORMALIZATION_SPEC


def resolve_stock_label_spec(value):
    """Resolve the explicitly versioned open(T+1)/close(T+h) family.

    Older R0 LabelSpec v1 measures end-start; it is not reinterpreted here.
    Metadata remains transport-only and does not change the label identity.
    """
    if type(value) is dict:
        wire=deepcopy(value); wire.setdefault('metadata',{})
        if type(wire.get('maturity')) is dict: wire['maturity'].setdefault('metadata',{})
        label=from_dict(wire)
    else: label=value
    require(type(label) is LabelSpec, 'typed stock LabelSpec required')
    label = validate(label, require_resolved=True)
    require(label.contract_version == '2' and label.key == ('security_id', 'session'),
            'stock target requires LabelSpec v2 and the complete security/session key')
    h = label.horizon_sessions
    require(type(h) is int and h > 0 and label.return_start_offset_sessions == 1 and
            label.return_end_offset_sessions == h, 'positive stock endpoint horizon required')
    require(label.formula in ('close(f+h) / open(f+1) - 1',
                             f'close(f+{h}) / open(f+1) - 1'),
            'unsupported stock return formula')
    require(label.price_basis == 'common_anchor_adjusted_v1',
            'stock target requires the declared common-anchor adjusted Data view')
    require(label.normalization_policy == 'none' and
            label.missing_delisting_policy == 'invalid_null_preserve_grid' and
            type(label.maturity) is MaturitySpec and label.maturity.lag_sessions == h,
            'unsupported stock normalization, missing policy or maturity')
    require(label.return_start_rule=='next_session_open' and label.return_end_rule=='horizon_session_close' and
        label.benchmark_semantics=='absolute_return' and
        label.corporate_action_semantics=='factor_ratio_no_separate_cashflow' and
        label.maturity.calendar_policy=='actual_exchange_sessions' and
        label.maturity.availability_rule=='max_endpoint_price_factor_anchor_usable_from',
        'unsupported executable stock Label policy')
    require(label.maturity.rule=='all_outcome_dependencies_strictly_before_fit_cutoff',
        'stock LabelSpec v2 supports strictly_before_fit_cutoff only; <= maturity is not supported')
    wire = to_dict(label)
    # Transport metadata is deliberately not part of the effective request.
    wire.pop('metadata', None)
    wire['maturity'].pop('metadata', None)
    return {'contract_version': 'stock_target_spec_v1', 'label_spec': wire,
            'label_definition_ref': semantic_identity(label), 'horizon_sessions': h,
            'target_semantics': f'forward_{h}_session_cs_zscore_prediction',
            'raw_label_id': f'forward_{h}_session_open_close_v1',
            'formula': f'close(f+{h}) / open(f+1) - 1',
            'normalization': deepcopy(NORMALIZATION_SPEC)}


def stock_label_wire(target, *, calendar, anchor):
    """Compile the neutral Engine Label ABI from the declared typed definition.

    The reusable definition ref and the scoped ABI label_spec_ref differ:
    the latter additionally includes this actual calendar and adjustment anchor.
    """
    from .stock_artifacts import digest
    label=target['label_spec'];maturity=label['maturity']
    return {'label_id':target['raw_label_id'],'semantic_version':'1',
        'horizon_sessions':target['horizon_sessions'],'start_session_offset':1,
        'end_session_offset':target['horizon_sessions'],'start_price':'open','end_price':'close',
        'calendar_ref':digest({'contract_version':'stock_label_calendar_v1','sessions':calendar}),
        'price_basis':label['price_basis'],'adjustment_anchor':anchor,'formula':'close(f+h) / open(f+1) - 1',
        'normalization':'none','costs':'none','corporate_action_policy':label['corporate_action_semantics'],
        'availability':maturity['availability_rule'],'maturity_rule':maturity['rule'],
        'missing_policy':label['missing_delisting_policy']}


def label_horizon(common):
    target = common.get('target_spec')
    return 5 if target is None else target['horizon_sessions']


def target_semantics(common):
    return f'forward_{label_horizon(common)}_session_cs_zscore_prediction'


def eligible_target_reason(raw,feature,width,cutoff,common):
    """Preserve v1 admission; apply the explicitly declared new clock policy."""
    from .stock_label_contracts import _eligible_reason,_instant
    reason=_eligible_reason(raw,feature,width,cutoff)
    target=common.get('target_spec')
    if reason is None and target is not None and target['label_spec']['maturity']['rule']=='all_outcome_dependencies_strictly_before_fit_cutoff':
        if _instant(raw['label_available_at'])>=cutoff:return 'LABEL_NOT_MATURE'
    return reason
