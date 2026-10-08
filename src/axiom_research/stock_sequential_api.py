"""Thin public contexts over the existing concrete Research owner/writer."""
from contextlib import contextmanager


@contextmanager
def open_stock_ml_batch_preparation(data,*,feature_inputs,fold_specs,destination,preparation_options,
    label_spec=None,column_source=None,model_feature_selection=None,reuse_raw_from_batch=None,metrics=None,progress=None):
    """Prepare one fold, build using owner.batch, then advance within one owner.

    owner.next_fold() returns one admitted saved input or None. owner.finish()
    publishes COMPLETE only after every declared fold is saved. No fit or
    prediction occurs in this context itself. A caller-supplied ColumnSource
    remains caller-owned; all Research selections are released on exit.
    """
    from .stock_compact_labels import _prepare_compact_incrementally
    with _prepare_compact_incrementally(data,feature_inputs=feature_inputs,fold_specs=fold_specs,
        destination=destination,preparation_options=preparation_options,label_spec=label_spec,
        column_source=column_source,model_feature_selection=model_feature_selection,
        reuse_raw_from_batch=reuse_raw_from_batch,metrics=metrics,progress=progress) as owner:
        yield owner


@contextmanager
def open_stock_signal_evaluation_freeze(batch,*,scope,destination,signal_name):
    """Save admitted OOS shards directly from each build's existing projection.

    Run builders with this batch while the context is active; writer.finish()
    returns an immutable saved input ref after the batch is COMPLETE. This
    performs no statistic computation or account execution.
    """
    from .stock_signal_evaluation_build import _freeze_build_oos_inputs
    with _freeze_build_oos_inputs(batch,scope=scope,destination=destination,signal_name=signal_name) as writer:
        yield writer
