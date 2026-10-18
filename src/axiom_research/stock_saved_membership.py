"""Read missing legacy membership flags from the existing archived rows."""
from collections import defaultdict
import numpy as np

from .stock_fold_inputs import require
from .stock_matrix_storage import instant_us
from .stock_matrix_feature_column_metadata import _physical, _saved_json


def _legacy_membership_masks(child, spec, universe_id, profile, member, clocks, clock_known, reasons):
    """Map explicit old reader rows; close pools against PIT-selected set hashes."""
    from axiom_data.reader import _row_availability, _revision_order
    from axiom_data.vendor_membership import _digest as member_set_digest
    days=child['sessions']; securities=spec['universe']; shape=(len(days),len(securities))
    context=child['context'];query=context['query'];coverage=context['coverage']
    require(child['contract_version']=='reader_column_evidence_v1' and child['fields']==['is_member'] and
            context['domain']=='universe_membership' and context['snapshot_id']==spec['snapshot'] and
            context['source_profile_id']==profile['id'] and
            query['universe_id']==universe_id and query['symbols']==securities and query['sessions']==days and
            query['pit_policy']==spec['pit_policy'] and query.get('policy_by_session') is None and
            query['purpose']=='decision_facts' and query['fields']==['is_member'] and
            all(query['cutoff_by_session'][d]==spec['cutoff_by_session'][d] for d in days),
            'legacy membership evidence scope/PIT mismatch')
    rows=child['field_meta']['is_member']['by_key']; keyed={(r['session'],r['security_id']):r for r in rows}
    require(len(keyed)==len(rows)==len(days)*len(securities) and
            set(keyed)=={(d,s) for d in days for s in securities},'legacy membership row coverage mismatch')
    ordered=[keyed[d,s] for d in days for s in securities]
    require(all('missing_reason' in r and r['missing_reason'] in reasons for r in ordered),
            'explicit legacy membership missing evidence required')
    known=np.array([r['missing_reason'] is None for r in ordered],dtype=bool).reshape(shape)
    codes=np.array([reasons.index(r['missing_reason']) for r in ordered],dtype='u1').reshape(shape)
    expected_clocks=np.array([instant_us(r['usable_from']) if r.get('usable_from') else 0 for r in ordered],dtype='<i8').reshape(shape)
    expected_known=np.array([r.get('usable_from') is not None for r in ordered],dtype=bool).reshape(shape)
    require(np.array_equal(clocks,expected_clocks) and np.array_equal(clock_known,expected_known) and
            not np.any(member & ~known) and not np.any(known & ~clock_known),
            'legacy membership value/clock evidence mismatch')
    require(all(r.get('availability_basis') in (None,'declared_vendor_assumption') and
            (r.get('usable_from') is None or instant_us(r['usable_from'])<=instant_us(spec['cutoff_by_session'][r['session']])) and
            (r.get('source_snapshot_date') is None or r['source_snapshot_date']<=r['session']) for r in ordered),
            'legacy membership exceeds historical cutoff')
    require(coverage.get('basis')=='tushare_dated_index_weight_snapshot_v1' and
            coverage.get('between_snapshots')=='carry_forward', 'unsupported legacy membership coverage')
    groups=defaultdict(list);releases={}
    for state in coverage.get('complete_states',[]):
        if state.get('universe_id')==universe_id:
            require(state.get('complete') is True and state.get('member_set_source')=='canonical_intervals_v1',
                    'explicit legacy complete state required')
            groups[state['state_id']].append(state)
    candidates=[]
    # This repair is confined to best_effort dated releases, whose visibility
    # is identical for revisions of the same snapshot. Use Data's ordering.
    for state_id,states in groups.items():
        require(len({s['source_snapshot_date'] for s in states})==1,'mixed snapshot state revisions')
        state=_revision_order(states,state_id,profile=profile)
        release,_=_row_availability(state,spec['pit_policy'],profile,state['source_snapshot_date'],releases)
        candidates.append((state,instant_us(release.isoformat())))
    complete=np.zeros(shape,dtype=bool)
    for i,day in enumerate(days):
        if day>coverage.get('verified_through',''):continue
        visible=[(s,t) for s,t in candidates if s['source_snapshot_date']<=day and t<=instant_us(spec['cutoff_by_session'][day])]
        if not visible:continue
        newest=max(s['source_snapshot_date'] for s,_ in visible)
        winners=[(s,t) for s,t in visible if s['source_snapshot_date']==newest]
        require(len(winners)==1,'ambiguous legacy membership snapshot')
        state,release=winners[0];selected=[s for j,s in enumerate(securities) if member[i,j] and known[i,j]]
        same_snapshot=all(keyed[day,s].get('source_snapshot_date')==newest for j,s in enumerate(securities) if known[i,j])
        closed=(same_snapshot and member_set_digest(selected)==state['member_set_sha256'])
        if closed:
            # No new clock may be smuggled in by the added complete-state proof.
            require(bool(np.all(clock_known[i])) and bool(np.all(clocks[i]>=release)),
                    'complete-state proof needs a later membership clock')
            complete[i,:]=True
    return known,complete,codes


class LegacyMembershipMasks:
    """One archived chunk at a time, shared by the three missing flag arrays."""
    names=('member_known','member_pool_complete','member_reasons')

    def __init__(self, manifest, arrays, store, universe_id):
        self.manifest,self.arrays,self.store,self.universe_id=manifest,arrays,store,universe_id
        self.shape=tuple(manifest['buffers']['member']['shape'])
        self.by_row=[];self.spans=[];offset=0
        for i,chunk in enumerate(manifest['chunks']):
            stop=offset+len(chunk['sessions']);self.spans.append((offset,stop))
            self.by_row.extend([i]*(stop-offset));offset=stop
        self.cached=None;self.masks=None;self.chunks_loaded=0

    def row(self, name, position):
        index=self.by_row[position];chunk=self.manifest['chunks'][index]
        desc=chunk['evidence'][-1]
        if self.cached!=index:
            child=self.store.read_json(desc,'evidence_ref')
            spec=self.manifest['spec'];begin,end=self.spans[index]
            require(child['sessions']==chunk['sessions']==spec['read_sessions'][begin:end],
                    'legacy membership chunk alignment mismatch')
            self.masks=_legacy_membership_masks(child,spec,self.universe_id,
                self.manifest['source_profiles']['universe_membership']['source_profile'],
                self.arrays['member'][begin:end],self.arrays['member_available'][begin:end],
                self.arrays['member_available_validity'][begin:end],self.manifest['reason_codes'])
            self.cached=index;self.chunks_loaded+=1
        else:self.store.check_path(desc['path'])
        return self.masks[self.names.index(name)][position-self.spans[index][0]]

    def buffer(self, name):
        return _LegacyMaskBuffer(self,name)


class _LegacyMaskBuffer:
    def __init__(self, owner, name):
        self.owner,self.name=owner,name;self.shape=owner.shape

    def __getitem__(self, key):
        rows,columns=key if isinstance(key,tuple) else (key,slice(None))
        indices=np.arange(self.shape[0])[rows]
        if np.ndim(indices)==0:
            result=self.owner.row(self.name,int(indices))[columns].copy()
        else:
            result=np.asarray([self.owner.row(self.name,int(i)).copy() for i in indices])[:,columns]
        result.flags.writeable=False
        return result
