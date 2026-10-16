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


def segmented_buffer(descriptors):
    """Append immutable date slabs without copying their numeric bytes."""
    from .stock_artifacts import digest
    from .stock_fold_inputs import require
    parts=[]
    for descriptor in descriptors:
        parts.extend(descriptor['segments'] if descriptor.get('layout')=='date_segments_v1' else [descriptor])
    require(bool(parts) and all(p['dtype']==parts[0]['dtype'] and p['shape'][1:]==parts[0]['shape'][1:] for p in parts),
            'date segments need identical fields and security axes')
    value={'layout':'date_segments_v1','dtype':parts[0]['dtype'],
           'shape':[sum(p['shape'][0] for p in parts),*parts[0]['shape'][1:]],'segments':parts}
    return {**value,'buffer_digest':digest(value)}


class SegmentedFeatureBuffer:
    """Array indexing over existing date slabs; only the requested slice is copied."""
    def __init__(self, descriptor, store, component=None):
        import numpy as np
        self.descriptor=descriptor;self.store=store;self.component=component
        self.shape=tuple(descriptor['shape'][:2] if component is not None else descriptor['shape'])
        self.dtype=np.dtype({'float64_le':'<f8','int64_le':'<i8','bool_u8':'bool','uint8':'u1'}[descriptor['dtype']])
        self.nbytes=int(np.prod(self.shape))*self.dtype.itemsize

    def __getitem__(self, key):
        import numpy as np
        key=key if isinstance(key,tuple) else (key,)
        key=(*key,*([slice(None)]*(len(self.shape)-len(key))))
        axes=[];projection=[]
        for size,selector in zip(self.shape,key):
            chosen=np.arange(size)[selector]
            if np.ndim(chosen)==0:axes.append(np.asarray([chosen]));projection.append(0)
            elif isinstance(selector,slice):axes.append(chosen);projection.append(slice(None))
            else:
                unique,inverse=np.unique(chosen,return_inverse=True)
                axes.append(unique);projection.append(inverse.reshape(chosen.shape))
        out=np.empty(tuple(len(a) for a in axes),dtype=self.dtype);start=0
        if self.descriptor['layout']=='security_segments_v1':
            for part in self.descriptor['segments']:
                locations={column:i for i,column in enumerate(part['columns'])}
                selected=np.asarray([i for i,column in enumerate(axes[1]) if int(column) in locations],dtype=int)
                if not selected.size:continue
                desc=part['buffer']
                value=(type(self)(desc,self.store,self.component) if 'layout' in desc else self.store.buffer(desc))
                if 'layout' not in desc and self.component is not None:value=value[:,:,self.component]
                source_columns=np.asarray([locations[int(axes[1][i])] for i in selected])
                out[np.ix_(np.arange(len(axes[0])),selected,*[np.arange(len(a)) for a in axes[2:]])]=value[np.ix_(axes[0],source_columns,*axes[2:])]
            out.flags.writeable=False
            return out[tuple(projection)]
        for desc in self.descriptor['segments']:
            end=start+desc['shape'][0];selected=np.flatnonzero((axes[0]>=start)&(axes[0]<end))
            if selected.size:
                value=(type(self)(desc,self.store,self.component) if 'layout' in desc else self.store.buffer(desc))
                if 'layout' not in desc and self.component is not None:value=value[:,:,self.component]
                elif 'layout' not in desc and desc['dtype']=='bool_u8':value=value.view('bool')
                out[np.ix_(selected,*[np.arange(len(a)) for a in axes[1:]])]=value[np.ix_(axes[0][selected]-start,*axes[1:])]
            start=end
        out.flags.writeable=False
        return out[tuple(projection)]

    def column(self, component):
        return type(self)(self.descriptor,self.store,component)


def security_segmented_buffer(descriptors, axes, securities):
    """Join disjoint prepared security slabs; retain all original files."""
    from .stock_artifacts import digest
    from .stock_fold_inputs import require
    positions={s:i for i,s in enumerate(securities)};segments=[]
    for descriptor,axis in zip(descriptors,axes):
        if descriptor.get('layout')=='security_segments_v1':
            segments.extend({'columns':[positions[axis[i]] for i in part['columns']], 'buffer':part['buffer']}
                            for part in descriptor['segments'])
        else:segments.append({'columns':[positions[s] for s in axis],'buffer':descriptor})
    selected=[i for part in segments for i in part['columns']]
    require(sorted(selected)==list(range(len(securities))),'security slabs must cover the exact disjoint axis')
    first=descriptors[0]
    require(all(d['dtype']==first['dtype'] and d['shape'][0]==first['shape'][0] and d['shape'][2:]==first['shape'][2:] for d in descriptors),'security slab shapes differ')
    value={'layout':'security_segments_v1','dtype':first['dtype'],
        'shape':[first['shape'][0],len(securities),*first['shape'][2:]],'segments':segments}
    return {**value,'buffer_digest':digest(value)}
