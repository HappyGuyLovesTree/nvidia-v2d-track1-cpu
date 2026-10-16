"""Actual full30 packing of distinct synthetic final layers; never production predictions."""
from pathlib import Path
import argparse, datetime, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import load, box_mesh
from r11_state import *
from r11_pack import pack_local
from r12_import import export_layers
from r12_budget import C2,TREE
from r12 import DEFAULT_INDEX

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    index=load(DEFAULT_INDEX);cfg=load(index['config']['path']);cfg={**cfg,'budget_parent':None}
    cfg['deadlines']={k:(datetime.datetime.now(TZ)+datetime.timedelta(minutes=10)).isoformat() for k in cfg['deadlines']}
    roster=load(cfg['assets']['roster']['path']);ledger=Ledger(out,cfg);writer=Writer(ledger)
    entries=[];code={'commit':'e'*40,'tree':'f'*40};man={'code':code}
    for row in roster['episodes']:
        e=row['episode'];ids=row['required_original_frames'];T=max(ids)+1;mask=np.zeros(T,bool);mask[ids]=True
        data={'pose':np.full((T,136),np.nan),'scales':np.full(68,1+e/1000),'shape':np.full(45,e/1000),
              'object_rotation':np.full((T,3,3),np.nan),'object_translation':np.full((T,3),np.nan),'object_scale':np.array(1.)}
        data['pose'][ids]=np.asarray(ids)[:,None]/10000+e/100;data['object_rotation'][ids]=np.eye(3);data['object_translation'][ids]=[e/100,0,1]
        vertices,faces=box_mesh(np.array([.1+e/1000,.04,.03]));d=out/'fixture_source'/f'{e:06d}'
        refs={'npz':writer.npz(d/'dense.npz','C',**data),'mask':writer.npz(d/'mask.npz','C',mask=mask,original_frames=np.array(ids)),
              'mesh':writer.npz(d/'mesh.npz','C',vertices=vertices,faces=faces)}
        text=''.join('v '+' '.join(format(float(x),'.17g') for x in v)+'\n' for v in vertices)+''.join('f '+' '.join(str(int(x)+1) for x in f)+'\n' for f in faces)
        refs['obj']=writer.bytes(d/'object.obj',text.encode(),'C')
        vr=writer.json(d/'validation.json',{'fixture_only':True,'episode':e})
        entries.append({'episode':e,'required':ids,'source_refs':refs,'validation_ref':vr,
                        'prediction_producer_commit':C2 if e<23 else code['commit'],'producer_tree':TREE if e<23 else code['tree'],
                        'importer_code':code if e<23 else {}})
    view,coverage,inputs=export_layers(out/'view',entries,writer,code)
    packed=pack_local(out/'view',cfg,man,coverage,inputs,writer,'fixture_full30')
    require(packed['rows']==740780 and packed['frame_count']==9877 and packed['episode_count']==30,'full30 fixture pack')
    require(len({r['numeric_producer'] for r in view['episodes']})==2,'mixed provenance lost')
    atomic_json(out/'result.json',{'schema':'T012.05_full30_pack_fixture_v1','created_at':now(),'fixture_only':True,'not_production_predictions':True,
                'passed':True,'rows':packed['rows'],'frame_count':packed['frame_count'],'episode_count':packed['episode_count'],
                'view_ref':ref(out/'view/view.json'),'pack_ref':ref(out/'view/packs/fixture_full30/validation.json'),
                'real_model_loads':0,'real_model_forwards':0,'real_video_decodes':0,'synthetic_layers_generated_distinctly_per_episode':True},True)
    print('passed: synthetic 23+7 view, exact 740780 sample keys, all30 roundtrip')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output.resolve())
