"""Exclusive evidence IO, exact array identity and frozen execution identity."""
from pathlib import Path
import hashlib, io, json, os, re, subprocess
import numpy as np

class ContractError(ValueError):
    pass

def need(ok, why):
    if not ok:
        raise ContractError(why)

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def reference(path):
    p=Path(path).resolve()
    need(p.is_file(), 'reference missing')
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def verify(r):
    p=Path(r['path'])
    need(p.is_absolute() and p.resolve()==p and not p.is_symlink(), 'noncanonical reference')
    need(p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'], 'reference bytes/hash differs')
    return p

def ref_matches(original, pinned):
    """Historical refs can omit bytes; preserve them and verify normalized stat separately."""
    return (set(original) in ({'path','sha256'},{'path','bytes','sha256'}) and
            original['path']==pinned['path'] and original['sha256']==pinned['sha256'] and
            ('bytes' not in original or original['bytes']==pinned['bytes']))

def write_new(path, value, readonly=False):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    need(not p.is_symlink() and p.parent.resolve()==p.parent, 'output symlink')
    with p.open('xb') as f:
        f.write(value if isinstance(value, bytes) else encoded(value)); f.flush(); os.fsync(f.fileno())
        if readonly: os.fchmod(f.fileno(), 0o444)
    return reference(p)

def save_npz(path, arrays):
    buf=io.BytesIO(); np.savez_compressed(buf, **arrays)
    return write_new(path, buf.getvalue(), True)

def bits_equal(a, b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')

def array_identity(a):
    return dict(shape=list(a.shape), dtype=a.dtype.str, bytes_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())

def arrays_identity(values):
    return digest({k:array_identity(v) for k,v in values.items()})

def legacy_array_hash(a):
    return digest(dict(shape=list(a.shape), dtype=str(a.dtype), bytes_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest()))

def legacy_arrays_hash(v):
    return digest({k:legacy_array_hash(a) for k,a in v.items()})

class Reader:
    """Every consumed data file must be a pinned member of the private contract."""
    def __init__(self, refs):
        self.refs={r['path']:r for r in refs}
        need(len(self.refs)==len(refs), 'duplicate whitelist path')
        self.used=set()
        self.verified=set(); self.json_cache={}; self.legacy_cache={}
    def path(self, r):
        need(self.refs.get(r['path'])==r, 'reference outside exact whitelist')
        p=Path(r['path'])
        if r['path'] not in self.verified: p=verify(r); self.verified.add(r['path'])
        self.used.add(r['path']); return p
    def json(self, r, fields=None):
        key=(r['path'],tuple(fields) if fields else None)
        if key not in self.json_cache:
            text=self.path(r).read_text()
            if fields:
                from .restricted_json import extract
                self.json_cache[key]=extract(text,fields)
            else:self.json_cache[key]=json.loads(text)
        return self.json_cache[key]
    def legacy_digest(self, r):
        if r['path'] not in self.legacy_cache:
            from .restricted_json import legacy_content_digest
            self.legacy_cache[r['path']]=legacy_content_digest(self.path(r))
        return self.legacy_cache[r['path']]
    def npz(self, r, keys=None, *, subset=False):
        with np.load(self.path(r), allow_pickle=False) as z:
            need(keys is None or (set(keys)<=set(z.files) if subset else set(keys)==set(z.files)), 'NPZ fields differ')
            a={k:z[k] for k in (keys if subset else z.files)}
        need(all(x.dtype.kind in 'bfiu' for x in a.values()), 'unsupported array dtype')
        return a
    def finish(self):
        for p in self.used: verify(self.refs[p])

def source_files(root):
    root=Path(root).resolve()
    return sorted(p for p in root.rglob('*.py') if not ({'.git','__pycache__'} & set(p.relative_to(root).parts)))

def source_freeze(root):
    root=Path(root).resolve(); fs=[reference(p) for p in source_files(root)]
    return dict(schema='t1_public_python_source_v2', scope='all_public_python_including_baseline_portable_and_integration',files=fs,
                source_tree_sha256=digest({str(Path(r['path']).relative_to(root)):r['sha256'] for r in fs}))

def verify_freeze(r, expected):
    frozen=json.loads(verify(r).read_text()); root=Path(__file__).resolve().parents[1]
    need(source_freeze(root)==frozen and frozen['source_tree_sha256']==expected, 'actual integration source differs')
    return frozen

def git_identity(root, commit, clean=True):
    root=Path(root).resolve(); env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    def git(*args): return subprocess.check_output(['git','-C',str(root),*args], env=env, text=True).strip()
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None and git('rev-parse','HEAD')==commit, 'actual Git HEAD mismatch')
    if clean: need(git('status','--porcelain')=='', 'production source must be clean')
    return dict(commit=commit, tree=git('rev-parse','HEAD^{tree}'))

def read_slice(r):
    p=Path(r['path']); need(p.is_absolute() and p.resolve()==p and not p.is_symlink(), 'journal path')
    need(type(r['offset']) is int and r['offset']>=0 and 0<r['length']<=16384, 'journal slice bounds')
    with p.open('rb') as f: f.seek(r['offset']); raw=f.read(r['length'])
    need(raw.endswith(b'\n') and hashlib.sha256(raw).hexdigest()==r['sha256'], 'journal slice mismatch')
    return json.loads(raw)
