"""Verify fixed R0 parent closures and retained T1 rig arrays; never run a model."""
import numpy as np
from .core import need, digest, bits_equal, array_identity, arrays_identity, legacy_array_hash, legacy_arrays_hash, read_slice,ref_matches

DENSE=('pose','scales','shape','object_rotation','object_translation','object_scale')
PARAMS=('q','shape','scales','t','R','scale','K')
SCORED=np.array([1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42])
F=np.array([1.,-1.,-1.])
POLICY=dict(method='T1_preserve_pose_depth_v1', q_z_factor=-10., object_z_factor=1.,
            rig_tolerance_m=1e-4, projection_max_px=2., min_depth_m=.05,
            changed_fields=['pose[frame,2]','object_translation[frame,2]'],
            evidence='persisted_native_rig_arrays', model_calls_full=0)

def verify_parent(reader, row, episode, source, frame, dense, allowed_slices):
    """Original production HMR/B/baseline+adapted rig output/input closure (F4)."""
    m=row['meta']; need((m['episode'],m['source'],m['frame'])==(episode,source,frame), 'parent frame/source')
    need(m['call_closed'] is True and m['dense_raw_sha256']==m['raw_sha256'], 'parent raw closure')
    raw=reader.npz(row['raw'], ['mhr_params','shape_params','cam_trans','joints_2d','joints_3d'])
    fr=reader.json(row['frame_receipt'],['/episode','/source','/frame','/raw','/hmr_outcome','/producer','/hmr_call_end','/input_identity/pts','/input_identity/time_base','/input_identity/hmr_tensor_hashes','/transform/K'])
    br=reader.json(row['b_receipt'],['/frame','/source','/output','/plan_hash','/hmr_fill','/call_ends'])
    bp=reader.json(row['b_plan'],['/producer','/shape','/scales','/parents'])
    b=reader.npz(row['b_params'], ['original_params','adapted_params','R','t'],subset=True)
    need(ref_matches(fr['raw'],row['raw']) and fr['raw']['sha256']==m['raw_sha256'], 'original raw identity')
    need(fr['hmr_outcome']=='hmr_valid' and (fr['episode'],fr['source'],fr['frame'])==(episode,source,m['raw_frame']), 'original HMR parent')
    need(fr['producer']==m['numeric_producer']==bp['producer'], 'numeric producer')
    need(br['frame']==frame and br['source']==source and ref_matches(br['output'],row['b_params']), 'B output identity')
    need(br['plan_hash']==reader.legacy_digest(row['b_plan']) and bp['parents'][str(frame)]==br['hmr_fill'], 'B plan lexical parent')
    need(br['hmr_fill']['frame']==m['raw_frame'] and br['hmr_fill']['raw_sha256']==m['raw_sha256'], 'B nearest parent')
    q=b['adapted_params'][:136].astype(dense['pose'].dtype)
    need(bits_equal(q,dense['pose'][frame]) and bits_equal(b['t'].astype(dense['object_translation'].dtype),dense['object_translation'][frame]), 'dense/B parent values')
    need(bits_equal(b['R'].astype(dense['object_rotation'].dtype),dense['object_rotation'][frame]), 'dense/B rotation')
    need(np.array_equal(np.asarray(bp['shape']),dense['shape']), 'R0 shape parent')
    refs=[row['hmr_call_end'],*row['rig_call_ends']]
    need(all(r in allowed_slices for r in refs) and len(row['rig_call_ends'])==2, 'exact parent slices')
    hm=read_slice(refs[0]); inp=fr['input_identity']
    need(fr['hmr_call_end']==refs[0] and hm['kind']=='call_end' and hm['model_kind']=='hmr' and hm['outcome']=='returned', 'HMR call end')
    need((hm['episode'],hm['frame'])==(episode,m['raw_frame']) and hm['output_identity']==legacy_arrays_hash(raw), 'HMR output binding')
    need(hm['input_identity']==digest(inp['hmr_tensor_hashes']), 'HMR tensor binding')
    need(br['call_ends']==row['rig_call_ends'], 'B rig slice identity')
    for role,key,r in zip(('baseline','adapted'),('original_params','adapted_params'),refs[1:]):
        e=read_slice(r); parent=digest(dict(params=legacy_array_hash(b[key]),shape=legacy_array_hash(np.asarray(bp['shape'],np.float32)[None]),plan=br['plan_hash']))
        need((e['episode'],e['frame'])==(episode,frame) and e['kind']=='call_end' and e['model_kind']=='rig' and e['outcome']=='returned', 'original rig closure')
        need(e['stage']=='B:'+role and e['input_identity']==parent, 'original rig input binding')
    if m['raw_frame']==frame:
        need(inp['pts']==m['pts'] and inp['time_base']==m['time_base'], 'original frame PTS')
    return m['raw_frame']==frame

def parameters(reader, r):
    p=reader.npz(r, PARAMS)
    for k,shape in [('q',(136,)),('shape',(45,)),('scales',(68,)),('t',(3,)),('R',(3,3)),('scale',()),('K',(3,3))]:
        need(p[k].shape==shape and np.isfinite(p[k]).all(), 'parameter schema/finite '+k)
    need(p['scale']>0, 'object scale')
    need(p['K'][0,0]>0 and p['K'][1,1]>0 and np.array_equal(p['K'][2],[0.,0.,1.]),'original K contract')
    return p

def retained_rig(reader, rig_ref, call_ref, params, binding, input_contract_sha=None, call_slices=None):
    c=reader.json(call_ref); v=reader.npz(rig_ref, ['vertices','joints'])
    need(c['schema']=='t1_persisted_rig_call_v1' and c['coordinate']=='mhr_native', 'rig call schema/coordinate')
    for k in ('run_id','policy_sha256','source_tree_sha256','model_sha256'):
        need(c['binding'][k]==binding[k], 'D call producer binding '+k)
    s,e=c['start'],c['end']
    need(s['event']=='start' and s['kind']=='rig' and e['event']=='end' and e['kind']=='rig', 'persisted call not closed')
    need(s['call_id'] and s['call_id']==e['call_id'] and e['output']==rig_ref and e['output_identity']==arrays_identity(v), 'rig output identity')
    need(all(s['parents'].get(k)==array_identity(params[k]) for k in ('q','shape','scales')), 'rig q/shape/scales input binding')
    p=np.concatenate([params['q'].astype(np.float32),params['scales'].astype(np.float32)]); sh=params['shape'].astype(np.float32)
    actual_D='raw_forward_ref' in c
    if actual_D:
        need(input_contract_sha is not None and s['parents']['input_contract_sha256']==input_contract_sha,'actual D input contract binding')
        matches=[x for x in (call_slices or []) if x['call_id']==s['call_id']]
        need(len(matches)==1, 'exact original D call slices required')
        actual_end=read_slice(matches[0]['end'])
        need(actual_end.get('kind')=='rig' and read_slice(matches[0]['start'])==s and actual_end==e,'canonical wrapper versus actual D journal kind/bytes')
        need(s['parents']['actual_params204']==array_identity(p) and s['parents']['actual_shape45']==array_identity(sh),'recorded actual float32 input identity')
        need(s['parents']['params_sha256']==legacy_array_hash(p) and s['parents']['shape_sha256']==legacy_array_hash(sh),'actual input legacy hash')
        for k in ('source_tree_sha256','policy_sha256','model_sha256'):need(s['parents'][k]==binding[k],'actual call producer/input parent')
        raw=reader.npz(c['raw_forward_ref'],['vertices','joints'])
        need(c['native_joint_selection']=='all127 first3 coordinates from raw[0,:,:3]' and c['camera_transform']=='F=diag(1,-1,-1); camera_m=native*F/100','actual D native/camera transform')
    else:
        forward=reader.npz(c['forward_inputs'],['params','shape'])
        need(forward['params'].dtype==forward['shape'].dtype==np.dtype('float32') and forward['params'].shape in ((204,),(1,204)) and forward['shape'].shape in ((45,),(1,45)), 'actual forward input schema')
        need(bits_equal(forward['params'].reshape(204),p) and bits_equal(forward['shape'].reshape(45),sh), 'stored versus actual forward dtype/input')
        need(c['forward_identity']=={k:array_identity(a) for k,a in forward.items()}, 'actual forward identity')
        raw=reader.npz(c['raw_forward'],['vertices','joints'])
        need(c['native_selection']=='vertices[0],joints[0,:,:3]' and c['raw_output_identity']==arrays_identity(raw),'raw output identity')
    need(raw['vertices'].shape==(1,18439,3) and raw['joints'].shape==(1,127,8), 'raw forward shape')
    need(bits_equal(raw['vertices'][0],v['vertices']) and bits_equal(raw['joints'][0,:,:3],v['joints']), 'native selection differs from complete raw forward')
    for k,sh in [('vertices',(18439,3)),('joints',(127,3))]:
        need(v[k].shape==sh and np.isfinite(v[k]).all(), 'retained rig schema/finite')
    return {k:a.astype(np.float64)*F/100. for k,a in v.items()}

def uv(points, K):
    need(np.isfinite(points).all() and np.all(points[:,2]>.05), 'projection depth/finite')
    h=points@K.T
    need(np.all(h[:,2]!=0), 'homogeneous projection depth')
    p=h[:,:2]/h[:,2,None]; need(np.isfinite(p).all(), 'projection finite')
    return p

def geometry_check(old, new, before, after, vertices, dz):
    """Recompute actual storage adaptation, rig/22 domain and full mesh projections."""
    expected={k:a.copy() for k,a in old.items()}
    if dz!=0.:
        expected['q'][2]=np.asarray(np.float64(old['q'][2])-10.*dz,dtype=old['q'].dtype)
        expected['t'][2]=np.asarray(np.float64(old['t'][2])+dz,dtype=old['t'].dtype)
    need(all(bits_equal(expected[k],new[k]) for k in PARAMS), 'non-T1 or incorrect dtype adaptation')
    if old['q'][2]==new['q'][2] and old['t'][2]==new['t'][2]: return dict(decision='inherit',reason='storage_no_change')
    delta=np.array([0.,0.,dz]); report={}
    for key,selection in [('vertices',slice(None)),('joints',SCORED)]:
        a,b=before[key][selection],after[key][selection]
        residual=float(np.abs(b-a-delta).max()); report[key+'_residual_m']=residual
    report['all127_diagnostic_m']=float(np.abs(after['joints']-before['joints']-delta).max())
    report['storage_delta_mismatch']=float(abs(-(float(new['q'][2])-float(old['q'][2]))/10.-(float(new['t'][2])-float(old['t'][2]))))
    if max(report['vertices_residual_m'],report['joints_residual_m'],report['storage_delta_mismatch'])>1e-4:
        return dict(report,decision='inherit',reason='rig')
    for name,a,b in [('body',before['vertices'],after['vertices']),('scored22',before['joints'][SCORED],after['joints'][SCORED]),
                     ('object',(vertices.astype(np.float64)*float(old['scale']))@old['R'].astype(np.float64).T+old['t'].astype(np.float64),(vertices.astype(np.float64)*float(new['scale']))@new['R'].astype(np.float64).T+new['t'].astype(np.float64))]:
        try: maximum=float(np.linalg.norm(uv(b,old['K'])-uv(a,old['K']),axis=1).max())
        except ValueError: return dict(report,decision='inherit',reason=name+'_depth_or_finite')
        report[name+'_max_px']=maximum
        if maximum>2.: return dict(report,decision='inherit',reason=name+'_projection')
    return dict(report,decision='accepted',reason='accepted')
