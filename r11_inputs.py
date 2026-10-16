"""Frozen per-episode inputs, exact source segments, audit set and source provenance."""
from pathlib import Path
from fractions import Fraction
import importlib, os, subprocess, sys
from common import load, sha
from tracking import digest, METHOD
from r11_state import require, ref, verify_ref, atomic_json, now
P=Path(__file__).resolve().parent
IDENTITY_FIELDS=('source_repository','revision','repository_path','declared_size_bytes','expected_sha256')

def git(*args):return subprocess.check_output(['git','-C',str(P),*args],text=True).strip()
def code_identity():
    require(not git('status','--porcelain'),'source repository must be clean')
    files={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    return {'commit':git('rev-parse','HEAD'),'tree':git('rev-parse','HEAD^{tree}'),'files':files}
def verify_code(man):
    require(code_identity()==man['code'],'source/config identity changed')
def segments(row):
    required=row['required_original_frames'];require(required==sorted(set(required)) and required,'invalid required set')
    mapping={};runs=[]
    for fid in required:
        if not runs or fid!=runs[-1][-1]+1:runs.append([])
        runs[-1].append(fid);mapping[fid]=len(runs)-1
    require([{'first':s[0],'last':s[-1],'count':len(s)} for s in runs]==row['required_frame_stretches'],'manifest stretches mismatch')
    return runs,mapping

def audit_set(roster):
    pairs=[]
    for row in roster['episodes']:
        e=row['episode']
        if e==16:frames=[50,62,63,70,81,254,289]
        else:frames=[f for s in segments(row)[0] for f in (s[0],s[-1])]
        require(set(frames)<=set(row['required_original_frames']),'audit frame outside required')
        pairs.extend((e,f) for f in frames)
    pairs=sorted(set(pairs));require(len(pairs)==135,'fixed audit set must be 135')
    return {'schema':'r11_fixed_audit_v1','pairs':[list(x) for x in pairs],'full_mesh_pairs':[[16,f] for f in (50,70,254,289)],'sha256':digest(pairs)}

def load_config(path):
    path=Path(path).resolve();cfg=load(path)
    require(cfg['method']==METHOD,'R011 config must explicitly select low-score method')
    for key in ('spec','adoption'):verify_ref(cfg[key])
    verify_ref(cfg['assets']['roster']);roster=load(cfg['assets']['roster']['path'])
    require(len(roster['episodes'])==30 and sum(len(r['required_original_frames']) for r in roster['episodes'])==9877,'unexpected fixed roster')
    return cfg,roster

def freeze_episode(row,cfg):
    e=row['episode'];latest=Path(cfg['data_evidence'])/f'episode{e:02d}_pts_latest.json'
    if not latest.exists():return None
    summary=load(latest)
    if summary.get('status')!='passed' or not summary.get('whole_pts_verified'):return None
    pts_ref={'path':summary['pts_path'],'sha256':summary['pts_sha256']};verify_ref(pts_ref);pts=load(pts_ref['path'])
    expected={k:row[k] for k in IDENTITY_FIELDS}
    require(pts['identity']==expected and pts['episode']==e,'per-episode data identity conflict')
    require(pts['manifest']['sha256']==cfg['assets']['roster']['sha256'],'PTS manifest conflict')
    require(pts['status']=='passed' and pts['whole_pts_verified'] and pts['exit_code']==0 and not pts['missing_required_frames'],'PTS not verified')
    f=pts['file_verification'];video=Path(pts['video_absolute_path']).resolve()
    require(f['file_verified'] and f['actual_sha256']==expected['expected_sha256'] and f['actual_size_bytes']==expected['declared_size_bytes'],'file verification conflict')
    require(video.stat().st_size==expected['declared_size_bytes'] and sha(video)==expected['expected_sha256'],'actual video conflict')
    require(pts['decoded_frame_count']==row['declared_frame_count']==len(pts['frame_map']),'wholePTS count mismatch')
    stream=pts['stream'];tb=Fraction(stream['time_base']);require(tb>0,'invalid stream time_base')
    require(stream['width']==row['declared_video']['video.width'] and stream['height']==row['declared_video']['video.height'],'dimensions conflict')
    previous=None
    for ordinal,m in enumerate(pts['frame_map']):
        require(m['original_frame']==ordinal and type(m['pts']) is int,'invalid wholePTS frame ordinal')
        require(previous is None or m['pts']>previous,'nonmonotonic wholePTS');previous=m['pts']
    require(pts['required_covered']==len(row['required_original_frames']),'required PTS coverage mismatch')
    freeze=load(cfg['data_code_freeze']['path']);verify_ref(cfg['data_code_freeze'])
    require(pts['code_hashes']==freeze['code_hashes'],'unfrozen PTS checker')
    for name,h in freeze['code_hashes'].items():require(sha(Path(freeze['code_directory'])/name)==h,'data checker source changed')
    runs,mapping=segments(row)
    return {'schema':'r11_episode_input_v1','episode':e,'source_identity':expected,'video':{'path':str(video),'sha256':expected['expected_sha256']},'whole_pts':pts_ref,'checker_hashes':pts['code_hashes'],'required':row['required_original_frames'],'sources':runs,'source_map':{str(k):v for k,v in mapping.items()},'stream':stream,'decoded_frame_count':pts['decoded_frame_count'],'mirror_declaration_match':True,'official_independent_binding':bool(pts['official_independent_binding']),'local_file_verified':True,'whole_pts_verified':True}

def verify_episode(ep,cfg):
    verify_ref(cfg['assets']['roster']);row=next(r for r in load(cfg['assets']['roster']['path'])['episodes'] if r['episode']==ep['episode'])
    require(ep['source_identity']=={k:row[k] for k in IDENTITY_FIELDS} and ep['required']==row['required_original_frames'],'episode changed against fixed manifest')
    runs,mapping=segments(row);require(ep['sources']==runs and ep['source_map']=={str(k):v for k,v in mapping.items()},'source segment map changed')
    require(ep['video']['sha256']==row['expected_sha256'],'video declaration changed')
    verify_ref(ep['video']);verify_ref(ep['whole_pts']);pts=load(ep['whole_pts']['path'])
    require(pts['identity']==ep['source_identity'] and pts['stream']==ep['stream'],'frozen episode contract changed')
    require(pts['code_hashes']==ep['checker_hashes'],'frozen checker changed')
    return pts

def runtime_identity(names):
    result={'python':sys.version,'executable':str(Path(sys.executable).resolve()),'environment_threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')},'provider':'CPUExecutionProvider','precision':'float32'}
    for name in names:
        mod=importlib.import_module(name);p=Path(mod.__file__).resolve()
        result[name]={'version':getattr(mod,'__version__',None),'file':str(p),'module_sha256':sha(p)}
    extensions={'numpy':'numpy._core._multiarray_umath','av':'av._core','onnxruntime':'onnxruntime.capi.onnxruntime_pybind11_state','torch':'torch._C'}
    for name in names:
        if name in extensions:
            m=importlib.import_module(extensions[name]);result[name]['extension']=ref(m.__file__)
        elif name=='cv2':
            mod=importlib.import_module(name);result[name]['build_sha256']=digest(mod.getBuildInformation())
            binaries=list(Path(mod.__file__).parent.glob('*.so'))
            result[name]['binaries']=[ref(p) for p in binaries]
    # Environment lock and actual module versions are both retained.
    return result

def numeric_identity(cfg,man,runtime,ep):
    return {'video_sha256':ep['video']['sha256'],'detector_sha256':cfg['assets']['nanodet']['sha256'],'postprocess_sha256':sha(P/'vendor/nanodet.py'),'runtime_sha256':digest(runtime),'whole_pts_sha256':ep['whole_pts']['sha256'],'source_identity_sha256':digest(ep['source_identity']),'producer_commit':man['code']['commit'],'preprocess_sha256':sha(P/'vendor/preprocess.py'),'hmr_sha256':cfg['assets']['hmr']['sha256'],'code_tree':man['code']['tree']}

def numerical_hash(cfg):
    return digest({'method':cfg['method'],'assets':{k:v['sha256'] for k,v in cfg['assets'].items() if k not in ('roster','sample')},'rules':cfg['numerical_contract']})
