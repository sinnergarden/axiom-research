"""Optional, per-call trusted root for saved SignalEval's nested readers."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_ROOT = ContextVar("axiom_saved_signal_read_root", default=None)


def checked_saved_path(path):
    root = _ROOT.get()
    if root is None:
        return Path(path)
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("saved Signal evaluation nested file escapes research_root")
    return resolved


@contextmanager
def saved_read_root(root):
    resolved = None if root is None else Path(root).resolve(strict=True)
    if resolved is not None and not resolved.is_dir():
        raise ValueError("research_root must be a directory")
    token = _ROOT.set(resolved)
    try:
        yield
    finally:
        _ROOT.reset(token)
