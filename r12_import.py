"""Completed-layer provenance and copy views. Never restores tracker/raw or calls models."""
from pathlib import Path
import numpy as np
from common import load, sha
from tracking import digest
from r11_state import require, ref, verify_ref, read_npz, now
from r11_inputs import segments, IDENTITY_FIELDS
from r11_pack import check_dense
from r12_budget import C2, TREE

ARRAYS={'pose','scales','shape','object_rotation','object_translation','object_scale'}

def verify_frozen_input(input_ref,row,roster_ref):
    ep=load(verify_ref(input_ref));require(ep['schema']=='r11_episode_input_v1','input schema')
    require(ep['episode']==row['episode'] and ep['required']==row['required_original_frames'],'input required identity')
    runs,mapping=segments(row)
    require(ep['sources']==runs and ep['source_map']=={str(k):v for k,v in mapping.items()},'source boundaries changed')
    identity={k:row[k] for k in IDENTITY_FIELDS}
    require(ep['source_identity']==identity and ep['video']['sha256']==row['expected_sha256'],'video source identity')
    pts=load(verify_ref(ep['whole_pts']));verification=pts['file_verification']
    require(pts['manifest']['sha256']==roster_ref['sha256'] and pts['identity']==identity,'PTS/roster binding')
    require(pts['episode']==ep['episode'] and pts['status']=='passed' and pts['whole_pts_verified'] and pts['exit_code']==0 and not pts['missing_required_frames'],'PTS not passed')
    require(verification['file_verified'] and verification['actual_sha256']==ep['video']['sha256'] and verification['actual_size_bytes']==row['declared_size_bytes'],'frozen video verification changed')
    require(Path(ep['video']['path']).stat().st_size==row['declared_size_bytes'],'video size changed')
    require(pts['stream']==ep['stream'] and pts['code_hashes']==ep['checker_hashes'],'PTS/input drift')
    require(pts['decoded_frame_count']==ep['decoded_frame_count']==row['declared_frame_count']==len(pts['frame_map']),'PTS count')
    prior=None
    for i,m in enumerate(pts['frame_map']):
        require(m['original_frame']==i and type(m['pts']) is int and (prior is None or m['pts']>prior),'PTS ordinal')
        prior=m['pts']
    return ep

def verify_layers(inputs,ids):
    for k in ('npz','mask','obj','mesh'):verify_ref(inputs[k])
    data=read_npz(inputs['npz']['path']);maskdata=read_npz(inputs['mask']['path']);mesh=read_npz(inputs['mesh']['path'])
    require(set(data)==ARRAYS,'must have exactly dense six arrays')
    require(set(maskdata)=={'mask','original_frames'} and maskdata['mask'].dtype==np.bool_,'mask schema/dtype')
    mask=maskdata['mask'];require(maskdata['original_frames'].tolist()==ids and len(mask)==max(ids)+1,'original mask frame identity')
    check_dense(data,ids,mask)
    require(np.max(np.abs(data['object_rotation'][ids].transpose(0,2,1)@data['object_rotation'][ids]-np.eye(3)))<=1e-6,'rotation compatibility')
    require(np.all(np.linalg.det(data['object_rotation'][ids])>0),'rotation reflection')
    vertices=np.asarray(mesh['vertices']);faces=np.asarray(mesh['faces'])
    require(vertices.shape==(8,3) and faces.shape==(12,3) and np.isfinite(vertices).all(),'mesh geometry')
    require(faces.dtype.kind in 'iu' and faces.min()>=0 and faces.max()<8,'mesh indices')
    lines=Path(inputs['obj']['path']).read_text().splitlines()
    vv=np.asarray([[float(x) for x in line.split()[1:]] for line in lines if line.startswith('v ')])
    ff=np.asarray([[int(x)-1 for x in line.split()[1:]] for line in lines if line.startswith('f ')])
    require(np.array_equal(vv,vertices) and np.array_equal(ff,faces),'OBJ is not matching metre mesh')
    return {'dense_six_arrays':'passed','mask_exact':'passed','obj_mesh_exact':'passed',
            'dimensions_m':(vertices.max(0)-vertices.min(0)).tolist(),'frame_count':len(ids)}

def legacy_entry(pipeline_row,cfg,index_ref,manifest_ref):
    e=pipeline_row['episode'];require(0<=e<=22,'legacy episode outside frozen23')
    roster_ref=cfg['assets']['roster'];roster=load(verify_ref(roster_ref));row=next(r for r in roster['episodes'] if r['episode']==e)
    ep=verify_frozen_input(pipeline_row['input'],row,roster_ref)
    pack_ref=pipeline_row['pack'];pack=load(verify_ref(pack_ref))
    c_ref=pipeline_row['stages']['C']['receipt'];c=load(verify_ref(c_ref))
    require(c['schema']=='r11_stage_receipt_v1' and c['status']=='completed' and c['stage']=='C' and c['episode']==e and c['producer']==C2,'legacy C receipt identity')
    require(c['details']['pack']==pack_ref and c['details']['frames']==ep['required'],'C/pack binding')
    require(pack['schema']=='r11_pack_receipt_v1' and pack['producer']==C2 and pack['numeric_index_mesh_pack']=='passed' and pack['completed']=={str(e):ep['required']},'legacy pack identity')
    inputs=pack['inputs'][str(e)];require(c['details']['inputs']==inputs and inputs['episode_input_sha256']==digest(ep),'pack/input binding')
    b=load(verify_ref(inputs['B_receipt']));plan=load(verify_ref(inputs['plan']))
    require(b['producer']==C2 and b['status']=='completed' and b['episode']==e and b['stage']=='B','B producer/identity')
    require(b['details']['plan']==inputs['plan'] and b['details']['mesh']==inputs['mesh'] and b['details']['rig_checked']==ep['required'],'B output binding')
    require(plan['producer']==C2 and plan['episode_input']==digest(ep) and set(map(int,plan['frames']))==set(ep['required']),'B plan binding')
    checks=verify_layers(inputs,ep['required'])
    require(np.allclose(checks['dimensions_m'],plan['dimensions'],rtol=0,atol=1e-12),'mesh dimensions/plan mismatch')
    return {'schema':'r12_imported_completed_v1','episode':e,'status':'imported_completed','imported':True,
       'prediction_producer_commit':C2,'producer_tree':TREE,'parent_index_ref':index_ref,'parent_manifest_ref':manifest_ref,
       'parent_config_ref':load(verify_ref(index_ref))['config'],'episode_input_ref':pipeline_row['input'],
       'whole_pts_ref':ep['whole_pts'],'roster_ref':roster_ref,'input_video_sha256':ep['video']['sha256'],
       'required':ep['required'],'required_digest':digest(ep['required']),'sources_digest':digest(ep['sources']),
       'source_refs':inputs,'C_receipt_ref':c_ref,'validation_ref':pack_ref,'validation':checks,
       'historical_C22_exit_reap':'unverified' if e==22 else 'not_applicable',
       'video_verification':'frozen_PTS_hash_chain_and_current_size_no_video_read','created_at':now()}

def copy_checked(source_ref,target,writer,category='C'):
    source=verify_ref(source_ref);target=Path(target)
    require(target.resolve()!=source.resolve(),'export must copy to distinct path')
    if target.exists():
        require(not target.is_symlink() and target.stat().st_ino!=source.stat().st_ino,'hardlink/symlink export forbidden')
        require(sha(target)==source_ref['sha256'],'existing copy changed')
    else:writer.bytes(target,source.read_bytes(),category,True)
    verify_ref(source_ref);r=ref(target);require(r['sha256']==source_ref['sha256'],'copy hash mismatch')
    require(target.stat().st_ino!=source.stat().st_ino,'hardlink export forbidden')
    return {**r,'bytes':target.stat().st_size}

def export_layers(root,entries,writer,code):
    root=Path(root);rows=[];inputs={};coverage={}
    require(len({e['episode'] for e in entries})==len(entries),'overlapping export episodes')
    for entry in sorted(entries,key=lambda e:e['episode']):
        e=entry['episode'];ids=entry['required'];verify_layers(entry['source_refs'],ids)
        paths={'npz':root/'episodes'/f'episode_{e:06d}.npz','obj':root/'episodes'/f'episode_{e:06d}_object.obj',
               'mask':root/'masks'/f'episode_{e:06d}.npz','mesh':root/'meshes'/f'episode_{e:06d}.npz'}
        copied={k:copy_checked(entry['source_refs'][k],p,writer) for k,p in paths.items()}
        # pack_local accepts Ref dictionaries with an optional bytes field.
        inputs[e]=copied;coverage[e]=ids
        rows.append({'episode':e,'numeric_producer':entry['prediction_producer_commit'],'producer_tree':entry['producer_tree'],
                     'importer_commit':entry.get('importer_code',{}).get('commit'),'controller_commit':code['commit'],
                     'exporter_commit':code['commit'],'required_digest':digest(ids),'source_refs':entry['source_refs'],
                     'export_refs':copied,'validation_ref':entry['validation_ref']})
    view={'schema':'r12_export_view_v1','created_at':now(),'episodes':rows,'submission_eligible':False,
          'completed_episodes':sorted(coverage),'frame_count':sum(map(len,coverage.values()))}
    writer.json(root/'view.json',view,'C',True)
    return view,coverage,inputs
