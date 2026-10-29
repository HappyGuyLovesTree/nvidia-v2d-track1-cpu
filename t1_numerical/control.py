"""Generic immutable IO; selected original bodies, no project paths."""
import json,hashlib,os,io
from pathlib import Path
import numpy as np
from geometry import need,ContractError

def encoded(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_new(path,value,readonly=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=value if isinstance(value,bytes) else encoded(value)
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
        if readonly:os.fchmod(f.fileno(),0o444)
    return reference(path)


def reference(path):
    path=Path(path);need(path.is_absolute() and path.resolve()==path and path.is_file(),'reference canonical file')
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def verify(ref):
    p=Path(ref['path']);need(p.is_absolute() and p.resolve()==p and not p.is_symlink(),'reference path changed')
    need(p.is_file() and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'reference bytes/hash mismatch')
    return p


def arrays_hash(arrays):
    return digest({k:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()) for k,a in arrays.items()})


def compact_digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def legacy_array_hash(a):return compact_digest(dict(shape=list(a.shape),dtype=str(a.dtype),bytes_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()))


def legacy_arrays_hash(v):return compact_digest({k:legacy_array_hash(a) for k,a in v.items()})

