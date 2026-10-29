"""Fixed z-only adapter and exact all-domain geometric rejection gates."""
import numpy as np
from d_common import need,POLICY,bit_equal

SCORED=np.asarray(POLICY['rig_scored22'],dtype=np.int64)

def adapt_z(old,target_z,cam):
    q,obj=old['pose'],old['object_translation']
    need(q.shape==(136,) and obj.shape==(3,) and cam.shape==(3,),'z adapter shape')
    need(q.dtype.kind==obj.dtype.kind=='f' and np.isfinite(q[:3]).all() and np.isfinite(obj).all() and np.isfinite(cam).all() and np.isfinite(target_z),'z adapter finite')
    new={k:v.copy() for k,v in old.items()}
    dz=float(target_z)-float(cam[2])
    if dz!=0.:
        new['pose'][2]=np.asarray(np.float64(q[2])-10.*dz,dtype=q.dtype)
        new['object_translation'][2]=np.asarray(np.float64(obj[2])+dz,dtype=obj.dtype)
    need(bit_equal(q[:2],new['pose'][:2]) and bit_equal(q[3:],new['pose'][3:]) and bit_equal(obj[:2],new['object_translation'][:2]),'z-only slot invariant')
    for k in set(old)-{'pose','object_translation'}:need(bit_equal(old[k],new[k]),'opaque bit invariant')
    changed=not(bit_equal(q,new['pose']) and bit_equal(obj,new['object_translation']))
    dq=float((np.float64(new['pose'][2])-np.float64(q[2]))/-10.)
    do=float(np.float64(new['object_translation'][2])-np.float64(obj[2]))
    return new,dict(changed=changed,reason='proposed' if changed else 'storage_no_change',ideal_delta_m=[0.,0.,dz],
                    actual_q_delta_m=[0.,0.,dq],actual_object_delta_m=[0.,0.,do],q_rounding_error_m=dq-dz,
                    object_rounding_error_m=do-dz,actual_human_object_increment_difference_m=abs(dq-do))

def params204(old):
    q=old['pose'];scales=old['scales'];shape=old['shape']
    need(q.shape==(136,) and scales.shape==(68,) and shape.shape==(45,) and all(np.isfinite(x).all() for x in (q,scales,shape)),'rig input shape/finite')
    p=np.concatenate((q.astype(np.float32),scales.astype(np.float32)))
    return p,shape.astype(np.float32)

def object_points(vertices,R,scale,t):
    need(vertices.ndim==2 and vertices.shape[1]==3 and len(vertices)>0 and R.shape==(3,3) and t.shape==(3,),'object transform domain')
    need(all(np.isfinite(a).all() for a in (vertices,R,t)) and np.shape(scale)==() and np.isfinite(scale) and scale>0,'object transform finite/scale')
    return (vertices.astype(np.float64)*float(scale))@R.astype(np.float64).T+t.astype(np.float64)

def projection_gate(before,after,K):
    need(before.shape==after.shape and before.ndim==2 and before.shape[1]==3 and len(before)>0,'projection domain shape')
    need(K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.array_equal(K[2],[0.,0.,1.]),'original K contract')
    if not(np.isfinite(before).all() and np.isfinite(after).all()):return dict(passed=False,reason='nonfinite_points',points=len(before))
    if not(np.all(before[:,2]>.05) and np.all(after[:,2]>.05)):return dict(passed=False,reason='nonpositive_depth',points=len(before))
    a=before@K.T;b=after@K.T
    uv_a=a[:,:2]/a[:,2:];uv_b=b[:,:2]/b[:,2:]
    if not(np.isfinite(uv_a).all() and np.isfinite(uv_b).all()):return dict(passed=False,reason='nonfinite_projection',points=len(before))
    displacement=np.linalg.norm(uv_b-uv_a,axis=1);maximum=float(displacement.max())
    return dict(passed=maximum<=2.,reason='passed' if maximum<=2. else 'projection_over_2px',points=len(before),max_displacement_px=maximum)

def rigid_gate(before,after,delta,adapter):
    for rig in (before,after):need(rig['vertices'].shape==(18439,3) and rig['joints'].shape==(127,3),'actual rig domains')
    if not all(np.isfinite(after[k]).all() and np.isfinite(before[k]).all() for k in ('vertices','joints')):
        return dict(passed=False,reason='rig_nonfinite')
    vr=np.abs(after['vertices']-before['vertices']-delta);jr=np.abs(after['joints']-before['joints']-delta)
    v,j=float(vr.max()),float(jr[SCORED].max());gap=adapter['actual_human_object_increment_difference_m']
    passed=max(v,j,gap)<=1e-4
    return dict(passed=passed,reason='passed' if passed else 'rig_translation_residual',mesh_max_abs_m=v,scored22_max_abs_m=j,all127_diagnostic_max_abs_m=float(jr.max()),actual_human_object_increment_difference_m=gap)

def all_gates(old,new,vertices,before,after,K,adapter):
    if not adapter['changed']:return dict(accepted=False,first_reason='storage_no_change',checks={})
    delta=np.asarray(adapter['ideal_delta_m'],np.float64)
    checks={'rig':rigid_gate(before,after,delta,adapter)}
    if not checks['rig']['passed']:return dict(accepted=False,first_reason=checks['rig']['reason'],checks=checks)
    checks['human_mesh']=projection_gate(before['vertices'],after['vertices'],K)
    checks['human_scored22']=projection_gate(before['joints'][SCORED],after['joints'][SCORED],K)
    checks['object_mesh']=projection_gate(object_points(vertices,old['object_rotation'],old['object_scale'],old['object_translation']),
                                           object_points(vertices,new['object_rotation'],new['object_scale'],new['object_translation']),K)
    for domain in ('human_mesh','human_scored22','object_mesh'):
        if not checks[domain]['passed']:return dict(accepted=False,first_reason=domain+':'+checks[domain]['reason'],checks=checks)
    return dict(accepted=True,first_reason='geometric_acceptance_only',checks=checks,visible_review='pending_independent')
