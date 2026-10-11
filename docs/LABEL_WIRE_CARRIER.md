# Label wire storage and bounded execution

This is a physical representation change. The logical Label metadata schema,
Raw Label formula, native metadata/selected-wire refs, Core cohort and output,
Feature ancestry, PIT clocks and fold selectors retain their existing checks.

`stock_native_json_carrier_v2` is accepted only for
`stock_matrix_label_metadata_v1`. It has the v1 fields plus exactly one
`wire_slots` entry at `['contents', selected_wire_ref]`. Its byte descriptor's
digest must equal that original contents key. The original child is strictly
`context`, `records`, `field_meta`; its canonical bytes are saved once and shared
by raw and normalized metadata. The physical skeleton child contains only the
context without coverage. It is never a validation credential. Existing v1 Raw,
Feature and Label files retain their interpretation.

Each fresh Store admits the real whole wire's canonical syntax, digest,
records/field_meta component refs and complete actual query key grid. Its
private observer uses the existing canonical scanner, with no new grammar.
Training retains context/coverage identity/component digests and compact key
coverage. It creates no records/provenance DOM. Evaluation decodes one cell
at a time, with the existing parent preflight reserving its text/object and
post-check allocations, and retains only requested endpoints. Those cells
still pass the original leaf/PIT/availability checks. File fingerprints bind
both the physical carrier and the body. Shared blobs are admitted once within
one Store; independent public loads verify their own actual bytes.

Raw rows are still present in Raw and raw/normalized metadata. Their control
parent remains subject to the explicit parent limit and decode reservation;
they have not been moved to an unbounded decoder. For a 64-session block the
row count is at most `64 * universe_width`. Maximum serialized row/control
size is data dependent; no real full-history size or RAM bound is claimed by
the small tests.

Preparation keeps its four required options and accepts optional
`maximum_source_bytes` and `maximum_parent_bytes`. Defaults remain 8 GiB source,
64 MiB parent and 512 MiB matrix/workspace. The explicit resident option is
mapped to matrix/workspace for Feature admission, publisher, complete staged
admission and prepared HIT. Public matrix batch limits accept the same optional
parent bound. Publisher counts unique written physical paths plus its already
admitted Feature closure before staged validation. This rejects cumulative
source growth earlier; complete staged validation still proves the whole saved
closure independently. RSS and wall-clock remain owned by the external monitor.

`tools/run_saved_fourfold.py` reads `AXIOM_FOURFOLD_PLAN`; it does not authorize
itself. Its physical counts come from the same date split rule used by prepare.
Counts change with storage blocks while the complete business row grid does
not. It accepts only complete, publicly validated prepared/fold HITs; unfinished
stages remain temporary and cannot resume. Finished folds retain original
definitions, byte/clock/grid checks and their public readback. Training counts
are zero for an admitted HIT, one for a fresh fold. A shared batch still serves
all four folds and closes before handoff.

Required next steps: independent correctness/efficiency review, then the
owner's smallest vertical synthetic Core check, then one real monitored run.
Pure byte/codec tests are not evidence that history preparation, four fits or
an Engine account completed. The interrupted 60-second legacy combination is
unconfirmed and must not be reported as a pass.
