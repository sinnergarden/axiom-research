"""Opt-in, block-committed staging for the existing physical input producer.

Only committed prefixes are reused. A crash before the atomic receipt replaces
itself leaves the last block uncommitted and safe to overwrite on recovery.
Staging is retained even after publication; no prior attempt is deleted here.
"""
import json
from hashlib import sha256
import os
from pathlib import Path

from .feature_input_buffers import allocate_feature_buffer
from .stock_artifacts import digest, file_digest, _verify_ref
from .stock_fold_inputs import require, seal


def atomic_json(path, value):
    from .feature_input_evidence import check_input_disk
    check_input_disk(Path(path).parent,len(json.dumps(value).encode())+1)
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


class FeatureInputCheckpoint:
    def __init__(self, root, *, binding, layout, sessions, metrics):
        import numpy as np
        self.root=Path(root).resolve(); self.path=self.root/'.feature-input-checkpoint.json'
        self.binding=digest(binding); self.layout=layout; self.sessions=sessions
        self.arrays={}
        if self.path.exists():
            state=json.loads(self.path.read_text()); _verify_ref(state,'checkpoint_ref')
            require(state['binding']==self.binding and state['layout']==layout,'input checkpoint binding/layout mismatch')
            done=state['completed_dates']; chunks=state['chunks']
            require(type(done) is int and 0<=done<=len(sessions) and
                    [d for c in chunks for d in c['sessions']]==sessions[:done], 'input checkpoint prefix mismatch')
            require(set(state['files'])==set(layout),'input checkpoint buffer set mismatch')
            for name,item in layout.items():
                path=Path(state['files'][name]); expected=int(np.prod(item['shape']))*np.dtype(item['dtype']).itemsize
                require(path.is_file() and path.stat().st_size==expected,'input checkpoint buffer missing/truncated')
                self.arrays[name]=np.memmap(path,mode='r+',dtype=item['dtype'],shape=tuple(item['shape']))
            cursor=0
            for block in state['buffer_blocks']:
                require(block['start']==cursor and cursor<block['stop']<=done,'input checkpoint block mismatch')
                require(set(block['digests'])==set(layout),'input checkpoint block buffers mismatch')
                for name,expected in block['digests'].items():
                    actual=sha256(memoryview(self.arrays[name][cursor:block['stop']]).cast('B')).hexdigest()
                    require(actual==expected,'input checkpoint committed buffer changed')
                cursor=block['stop']
            require(cursor==done,'input checkpoint committed prefix mismatch')
            for chunk in chunks:
                for evidence in chunk['evidence']:
                    path=Path(evidence['path'])
                    require(path.is_file() and file_digest(path)==evidence['file_digest'],'input checkpoint evidence missing/changed')
                    from .feature_input_evidence import shared_evidence_descriptors
                    for item in shared_evidence_descriptors(json.loads(path.read_text())):
                        proof=Path(item['buffer']['path'])
                        require(proof.is_file() and not proof.is_symlink() and file_digest(proof)==item['buffer']['file_digest'],'checkpoint shared evidence missing/changed')
            self.state={k:v for k,v in state.items() if k!='checkpoint_ref'}
        else:
            self.arrays={name:allocate_feature_buffer(self.root,dtype=item['dtype'],shape=tuple(item['shape']))
                         for name,item in layout.items()}
            self.state={'binding':self.binding,'layout':layout,
                        'files':{name:str(Path(value.filename).resolve()) for name,value in self.arrays.items()},
                        'completed_dates':0,'chunks':[],'buffer_blocks':[],'metrics':metrics}
            self.commit(0,[],metrics)

    def commit(self, completed, chunks, metrics):
        from .feature_input_evidence import check_input_disk
        check_input_disk(self.root)
        # The receipt is durable only after all numeric writes and evidence files.
        for array in self.arrays.values():
            array.flush()
            with open(array.filename,'rb') as stream: os.fsync(stream.fileno())
        from .feature_input_evidence import shared_evidence_descriptors
        synced=set()
        for chunk in chunks[len(self.state['chunks']):]:
            for item in chunk['evidence']:
                with open(item['path'],'rb') as stream: os.fsync(stream.fileno())
                for proof in shared_evidence_descriptors(json.loads(Path(item['path']).read_text())):
                    path=proof['buffer']['path']
                    if path not in synced:
                        with open(path,'rb') as stream: os.fsync(stream.fileno())
                        synced.add(path)
        blocks=list(self.state['buffer_blocks']); start=self.state['completed_dates']
        if completed>start:
            blocks.append({'start':start,'stop':completed,'digests':{name:
                sha256(memoryview(array[start:completed]).cast('B')).hexdigest() for name,array in self.arrays.items()}})
        self.state={**self.state,'completed_dates':completed,'chunks':list(chunks),'buffer_blocks':blocks,'metrics':dict(metrics)}
        atomic_json(self.path,seal(self.state,'checkpoint_ref'))


def extension_checkpoint(root, *, source, fields, kinds, sessions, block_sessions, binding, metrics):
    """Reuse the physical producer's existing committed-prefix receipt for extensions."""
    import numpy as np
    layout={field+'::'+kind:{'dtype':np.dtype(dtype).str,
            'shape':[len(sessions),len(source.manifest['spec']['universe'])]}
            for field in fields for kind,(dtype,_) in kinds.items()}
    checkpoint=FeatureInputCheckpoint(root,binding={'source_ref':source.identity,'extension':binding,
        'block_sessions':block_sessions,'evidence_encoding':'reader_column_evidence_v2'},
        layout=layout,sessions=sessions,metrics=metrics)
    return checkpoint,{**source.arrays,**checkpoint.arrays}
