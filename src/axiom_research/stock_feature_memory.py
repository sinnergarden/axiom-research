"""Stage charges for the ordinary Reader -> public price adjustment path.

This models Data e457's fixed OHLC/factor path, not arbitrary transformations.
It does not execute Data, copy provenance, or materialize future output rows.
The charges cover Python graphs and frame buffers; external Reader/Arrow/native
allocations still belong to the caller's budget and process RSS observation.
"""
import struct
import sys


_POINTER = struct.calcsize("P")
_SCALARS = (type(None), bool, int, float, str)
_OHLC = ("open", "high", "low", "close")


def _list_bytes(count):
    # CPython append capacity, including its next growth table.
    return sys.getsizeof([]) + _POINTER * (count + count // 8 + 8)


def _index_bytes(count):
    # A generic-key dict uses hash/key/value entries. A set uses key/hash slots.
    slots = 8
    while (2 * slots) // 3 < count:
        slots *= 2
    index_width = 1 if slots < 256 else 2 if slots < 65536 else 4 if slots < 2**32 else 8
    mapping = sys.getsizeof({}) + slots * index_width + (2 * slots // 3) * 3 * _POINTER
    set_slots = 8
    while set_slots < max(8, 4 * count):
        set_slots *= 2
    return mapping, sys.getsizeof(set()) + set_slots * 2 * _POINTER


def _containers(value):
    """JSON-safe copying creates containers, while immutable scalars are borrowed."""
    if type(value) in _SCALARS:
        return 0
    if type(value) is dict:
        return sys.getsizeof(value) + sum(_containers(v) for v in value.values())
    if type(value) in (list, tuple):
        return max(sys.getsizeof(value), _list_bytes(len(value))) + sum(_containers(v) for v in value)
    raise ValueError("Feature adjustment budget expects ordinary JSON-shaped Reader metadata")


def _leaf_sizes(definition):
    items = definition["by_key"]
    total, maximum = 0, 0
    for item in items:
        if type(item) is not dict or any(type(v) not in _SCALARS for v in item.values()):
            raise ValueError("Feature adjustment budget requires unadjusted scalar Reader provenance")
        size = sys.getsizeof(item)
        total += size
        maximum = max(maximum, size)
    return total, maximum


def adjustment_peak_bytes(prices, factors, native_wire):
    """Bound the existing adjust/to_json/join phases over complete Reader inputs.

    All four OHLC provenance lists and the factor list must describe the normal
    complete Reader grid; Data still performs its own query/PIT/key validation.
    ``native_wire`` is this call's private projected price snapshot. The input
    graph is measured once across all roots, so its shared metadata is charged
    once. New provenance dictionaries are charged by their actual container
    sizes; the immutable source strings/numbers they borrow are not copied.
    Inputs remain charged for every phase, even if the caller releases them
    earlier. No coefficient multiplying the whole input graph is used.
    """
    from .stock_matrix_feature_producer import _owned_bytes

    count, factor_count = len(prices.frame), len(factors.frame)
    if any(len(prices.field_meta[f]["by_key"]) != count for f in _OHLC) or len(factors.field_meta["factor"]["by_key"]) != factor_count:
        raise ValueError("Feature adjustment budget requires complete Reader provenance")
    seen = set()
    roots = [prices.field_meta, prices.context, factors.field_meta, factors.context, native_wire]
    graph = _owned_bytes(roots, seen)
    # The walk owns ID integers and, briefly, old/new set tables during growth.
    # Ordinary Reader/native JSON has shallow nesting, so reserve 16 KiB for
    # the depth-first iterator stack rather than one iterator per visited item.
    walk_workspace = 2 * sys.getsizeof(seen) + len(seen) * sys.getsizeof(id(seen)) + 16 * 1024
    del seen
    frame_bytes = sum(int(batch.frame.memory_usage(deep=True).sum()) +
        object.__sizeof__(batch.frame) + sys.getsizeof(batch.frame.__dict__) +
        sys.getsizeof(batch.frame.columns) + sys.getsizeof(batch) for batch in (prices, factors))
    owned = graph + frame_bytes
    price_leaves = sum(_leaf_sizes(prices.field_meta[f])[0] for f in _OHLC)
    factor_leaves, factor_max = _leaf_sizes(factors.field_meta["factor"])
    if count and not factor_count:
        raise ValueError("Feature adjustment budget requires anchor factor rows")
    outer = sys.getsizeof(dict.fromkeys(("security_id", "session", "missing_reason",
        "price_provenance", "factor_provenance", "anchor_factor_provenance")))
    # Four outer records, four price leaves, and four factor/anchor leaves each.
    provenance = 4 * count * outer + price_leaves + 8 * count * factor_max
    metadata = (provenance + 4 * _list_bytes(count) +
        4 * sys.getsizeof(dict.fromkeys(("dtype", "unit", "recipe_version", "by_key"))) +
        sys.getsizeof(dict.fromkeys(_OHLC)))
    price_dict, price_set = _index_bytes(count)
    factor_dict, factor_set = _index_bytes(factor_count)
    tuple_bytes = sys.getsizeof((None, None))
    # _index owns row maps/keys; _meta owns five additional key indexes. Include
    # one complete expected-grid set while either input index is constructed.
    indexes = (6 * price_dict + factor_dict + 2 * price_set + 2 * factor_set +
        (6 * count + 2 * factor_count + max(count, factor_count)) * tuple_bytes +
        max(price_dict + price_set, factor_dict + factor_set))
    input_records = sum(len(batch.frame) * (sys.getsizeof(dict.fromkeys(batch.frame.columns)) +
        max(0, len(batch.frame.columns)-2) * sys.getsizeof(2**64-1)) + _list_bytes(len(batch.frame))
        for batch in (prices, factors))
    row = sys.getsizeof(dict.fromkeys(("security_id", "session", *_OHLC)))
    output_records = count * (row + 4 * sys.getsizeof(0.0)) + _list_bytes(count)
    # Public adjustment detaches context, then keeps independent input queries.
    context = (_containers(prices.context) + _containers(prices.context["query"]) +
        _containers(factors.context["query"]) +
        sys.getsizeof(dict.fromkeys(range(14))) +
        _list_bytes(len(prices.context.get("limitations", ())) + len(factors.context.get("limitations", ())) + 1))
    # Frame construction may hold its input records and multiple column buffers.
    frame_construction = 3 * int(prices.frame.memory_usage(deep=True).sum())
    adjusted = metadata + context + frame_construction
    adjustment = indexes + input_records + output_records + adjusted
    # to_json keeps adjusted metadata while copying its container tree, and
    # briefly keeps DataFrame records beside their JSON-safe shallow copies.
    serialization = adjusted + metadata + context + 2 * output_records
    # The keyed amount join adds tuple keys/indexes and one shallow output row.
    # Its new field/context/query headers and projection/digest strings also
    # remain beside the adjusted wire; most nested provenance is shared.
    join_headers = (sys.getsizeof(dict.fromkeys((*_OHLC, "amount_cny"))) +
        sys.getsizeof(dict.fromkeys(range(len(prices.context) + 2))) +
        sys.getsizeof(dict.fromkeys(prices.context["query"])) + _list_bytes(5) +
        sys.getsizeof(dict.fromkeys(("operation", "adjusted_ref", "native_ref", "native_fields"))) +
        _list_bytes(1) + 2 * sys.getsizeof("sha256:" + "0" * 64) +
        sys.getsizeof("keyed_join_native_amount"))
    join = (metadata + context + output_records + 2 * price_dict + 2 * price_set +
        4 * count * tuple_bytes + count * sys.getsizeof(dict.fromkeys(("security_id", "session", *_OHLC, "amount_cny"))) + _list_bytes(count) + join_headers)
    temporary = max(walk_workspace, adjustment, serialization, join)
    return {"owned_bytes": owned, "temporary_bytes": temporary, "peak_bytes": owned+temporary,
        "stages": {"measurement_bytes": walk_workspace, "adjustment_bytes": adjustment,
                   "serialization_bytes": serialization, "join_bytes": join},
        "charges": {"input_frame_bytes": frame_bytes, "shared_input_graph_bytes": graph,
                    "provenance_container_bytes": provenance, "metadata_container_bytes": metadata,
                    "input_index_bytes": indexes, "factor_leaf_bytes": factor_leaves},
        "row_count": count}
