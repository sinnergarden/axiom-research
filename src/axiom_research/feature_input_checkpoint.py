"""Opt-in, block-committed staging for the existing physical input producer.

Only committed prefixes are reused. A crash before the atomic receipt replaces
itself leaves the last block uncommitted and safe to overwrite on recovery.
After durable publication only this checkpoint's unreferenced staging retires.
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
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    check_input_disk(path.parent,len(json.dumps(value).encode())+1)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


class FeatureInputCheckpoint:
    def __init__(self, root, *, binding, layout, sessions, metrics, axis=0):
        import numpy as np
        self.root=Path(root).resolve(); self.path=self.root/'.feature-input-checkpoint.json'
        self.binding=digest(binding); self.layout=layout; self.sessions=sessions
        require(axis in (0,1),'input checkpoint axis required');self.axis=axis
        self.arrays={}
        if self.path.exists():
            state=json.loads(self.path.read_text()); _verify_ref(state,'checkpoint_ref')
            require(state.get('status')!='PUBLISHED','input already published; open its source.json instead of staging')
            require(state['binding']==self.binding and state['layout']==layout and state.get('axis',0)==axis,'input checkpoint binding/layout mismatch')
            done=state['completed_dates']; chunks=state['chunks']
            require(type(done) is int and 0<=done<=len(sessions) and
                    (axis==1 or [d for c in chunks for d in c['sessions']]==sessions[:done]), 'input checkpoint prefix mismatch')
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
                    actual=self._block_digest(self.arrays[name],cursor,block['stop'])
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
                        'axis':axis,
                        'files':{name:str(Path(value.filename).resolve()) for name,value in self.arrays.items()},
                        'completed_dates':0,'chunks':[],'buffer_blocks':[],'metrics':metrics}
            self.commit(0,[],metrics)

    def _block_digest(self,array,start,stop):
        if self.axis==0:return sha256(memoryview(array[start:stop]).cast('B')).hexdigest()
        value=sha256()
        for row in array:value.update(memoryview(row[start:stop]).cast('B'))
        return value.hexdigest()

    def commit(self, completed, chunks, metrics):
        from .feature_input_evidence import check_input_disk
        check_input_disk(self.root)
        # The receipt is durable only after all numeric writes and evidence files.
        for array in self.arrays.values():
            array.flush()
            with open(array.filename,'rb') as stream: os.fsync(stream.fileno())
        from .feature_input_evidence import shared_evidence_descriptors
        synced=set(); directories=set()
        for chunk in chunks[len(self.state['chunks']):]:
            for item in chunk['evidence']:
                with open(item['path'],'rb') as stream: os.fsync(stream.fileno())
                directories.add(Path(item['path']).parent)
                for proof in shared_evidence_descriptors(json.loads(Path(item['path']).read_text())):
                    path=proof['buffer']['path']
                    if path not in synced:
                        with open(path,'rb') as stream: os.fsync(stream.fileno())
                        synced.add(path)
                        directories.add(Path(path).parent)
        for directory in directories:
            fd=os.open(directory,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        blocks=list(self.state['buffer_blocks']); start=self.state['completed_dates']
        if completed>start:
            blocks.append({'start':start,'stop':completed,'digests':{name:
                self._block_digest(array,start,completed) for name,array in self.arrays.items()}})
        self.state={**self.state,'completed_dates':completed,'chunks':list(chunks),'buffer_blocks':blocks,'metrics':dict(metrics)}
        atomic_json(self.path,seal(self.state,'checkpoint_ref'))

    def retire(self, manifest):
        """Release this writer's staging after a durable complete source exists.

        Never traverse/delete inherited buffers or evidence. A crash leaves
        either a resumable checkpoint or a published source and redundant
        staging; it never removes the only published representation.
        """
        published=json.loads((self.root/'source.json').read_text())
        _verify_ref(published,'source_ref')
        require(published['source_ref']==manifest['source_ref'] and
                self.state['completed_dates']==len(self.sessions),
                'staging retirement requires complete durable publication')
        referenced=set()
        def visit(value):
            if isinstance(value,dict):
                if isinstance(value.get('path'),str): referenced.add(Path(value['path']).resolve())
                for item in value.values(): visit(item)
            elif isinstance(value,list):
                for item in value: visit(item)
        visit(published)
        # The codec atomically exposes immutable files, but retirement also
        # needs their bytes durable before removing the recovery copy.
        directories=set()
        for path in referenced:
            if path.is_relative_to(self.root) and path.is_file():
                with path.open('rb') as stream: os.fsync(stream.fileno())
                directories.add(path.parent)
        for directory in directories:
            fd=os.open(directory,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        files={Path(p).absolute() for p in self.state['files'].values()}
        for path in files:
            require(path.parent==self.root and path.name.startswith('.feature-input-') and
                    not path.is_symlink() and path.resolve() not in referenced,
                    'retirement path is not private unreferenced staging')
        for value in self.arrays.values():
            if Path(value.filename).absolute() in files:
                value.flush(); value._mmap.close()
        self.state={**self.state,'status':'PUBLISHED','published_source_ref':manifest['source_ref']}
        atomic_json(self.path,seal(self.state,'checkpoint_ref'))
        for path in files: path.unlink(missing_ok=True)


def extension_checkpoint(root, *, source, fields, kinds, sessions, block_sessions, binding, metrics):
    """Reuse the physical producer's existing committed-prefix receipt for extensions."""
    import numpy as np
    layout={field+'::'+kind:{'dtype':np.dtype(dtype).str,
            'shape':[len(sessions),len(source.manifest['spec']['universe'])]}
            for field in fields for kind,(dtype,_) in kinds.items()}
    checkpoint=FeatureInputCheckpoint(root,binding={'source_ref':source.identity,'extension':binding,
        'block_sessions':block_sessions,'evidence_encoding':'reader_column_block_v1'},
        layout=layout,sessions=sessions,metrics=metrics)
    return checkpoint,{**source.arrays,**checkpoint.arrays}
