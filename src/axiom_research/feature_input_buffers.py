"""Disk-backed staging for the existing immutable column buffer format."""
from pathlib import Path
import tempfile

from .stock_matrix_storage import write_buffer


def allocate_feature_buffer(root, *, dtype, shape):
    import numpy as np
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    from .feature_input_evidence import check_input_disk
    check_input_disk(root,int(np.prod(shape))*np.dtype(dtype).itemsize)
    with tempfile.NamedTemporaryFile(prefix='.feature-input-',dir=root,delete=False) as stream:
        path=Path(stream.name)
    value=np.memmap(path,mode='w+',dtype=dtype,shape=shape)
    value._feature_staging_path=path
    return value


def seal_feature_buffer(root, value, *, dtype, shape, retain_staging=False):
    """Publish with the shared codec, then release writable staging storage."""
    import numpy as np
    value.flush()
    wire=value.view('u1') if dtype=='bool_u8' else value
    descriptor=write_buffer(root,wire.reshape(-1),dtype=dtype,shape=list(shape))
    path=Path(value.filename)
    result=np.memmap(descriptor['path'],mode='r',dtype=value.dtype,shape=shape)
    del wire
    if not retain_staging:
        value._mmap.close()
        path.unlink()
    return descriptor,result
