"""Explicit aliases for sealed D schema. No IO, filtering, inference or producer relabeling."""
import copy
from .core import need

REASONS={'geometric_acceptance_only':'accepted','storage_no_change':'storage_no_change','rig_translation_residual':'rig','rig_nonfinite':'rig'}
for src,dst in [('human_mesh','body'),('human_scored22','scored22'),('object_mesh','object')]:
    REASONS[src+':projection_over_2px']=dst+'_projection'
    REASONS[src+':nonpositive_depth']=dst+'_depth_or_finite'
    REASONS[src+':nonfinite_points']=dst+'_depth_or_finite'
    REASONS[src+':nonfinite_projection']=dst+'_depth_or_finite'

def normalize_geometry(native,context):
    """Context is separately pinned D-plan/R0 metadata, never inferred from key_id text."""
    need(native['schema']=='t1_persisted_geometry_v1' and native['coordinate']=='mhr_native','sealed D schema/coordinate')
    need(context['schema']=='t1_D_adapter_context_v1' and context['binding']==native['binding'],'adapter actual D binding')
    catalog={x['key_id']:x for x in context['reference_catalog']}
    need(len(catalog)==len(context['reference_catalog']),'duplicate D plan key')
    refs=[]; normalized=[]
    for b in native['all_old_reference_rig_outputs']:
        k=b['parents']['key_id']; need(k in catalog and b['parents']['role']=='R0_reference','old rig D plan mapping')
        x=catalog[k]
        refs.append(dict(episode=x['episode'],source=x['source'],frame=x['frame'],role='R0_reference',binding=native['binding'],params_ref=x['params_ref'],rig_ref=b['output'],call_ref=b['call_receipt']))
    for x in native['rows']:
        need(x['key_id'] in catalog,'proposal D plan key absent'); meta=catalog[x['key_id']]
        need((x['episode'],x['source'],x['frame'])==(meta['episode'],meta['source'],meta['frame']),'D plan coordinate conflict')
        need(x['old_params']==meta['params_ref'],'D old parameter snapshot binding')
        mb=x['mesh_binding']
        need(mb['coordinate']=='local_metres' and mb['transform']=='(vertices*scale)@R.T+t; original scale/R and actual stored old/candidate t','actual D object transform semantics')
        need(mb['old_transform_params']==x['old_params'] and mb['candidate_transform_params']==x['candidate_params'],'D object parameter refs')
        need(x['reason'] in REASONS and x['decision']==('accepted' if x['gate']['accepted'] else 'inherit_R0'),'actual D gate/decision schema')
        y=copy.deepcopy(x);y.update(decision='accepted' if x['decision']=='accepted' else 'inherit',reason=REASONS[x['reason']],original_D_reason=x['reason'],object_mesh=dict(obj=mb['obj_ref'],mesh_npz=mb['mesh_npz_ref'],vertices_identity=mb['vertices_identity'],faces_identity=mb['faces_identity'],transform='(V*s)@R.T+t'))
        for k in ('pts','time_base','input_version'):y[k]=meta[k]
        normalized.append(y)
        refs.append(dict(episode=x['episode'],source=x['source'],frame=x['frame'],role='T1_candidate',binding=native['binding'],params_ref=x['candidate_params'],rig_ref=x['new_rig'],call_ref=x['new_call']))
    result=copy.deepcopy(native);result.update(rows=normalized,reference_rigs=refs,model_ref=context['model_ref'],source_freeze=context['source_freeze'])
    result['input_contract_sha256']=context['input_contract']['sha256']
    result['call_slices']=context.get('call_slices',[])
    need(len({x['call_id'] for x in result['call_slices']})==len(result['call_slices']),'duplicate D call slice identity')
    result['normalization']='field aliases only; actual D bindings and original reasons retained'
    return result
