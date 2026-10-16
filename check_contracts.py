"""Synthetic contract checks. Does not instantiate any model or invoke forward."""
from pathlib import Path
import argparse,ast,inspect,json,sys
import numpy as np
from common import *

def main(out):
 cases=[]
 def c(i,b,s=.9):return candidate(i,b,s,0,100,100)
 t=Tracker();r=t.step(50,[],100,100);assert r['branch']=='full_no_detection' and r['last_observed_bbox'] is None;cases.append('zero_detection_empty_state')
 t=Tracker();r=t.step(50,[c(1,[20,20,40,40]),c(0,[10,10,30,30])],100,100);assert r['selected_index']==0;cases.append('multi_equal_score_area_coordinate_tie')
 t=Tracker();r=t.step(50,[c(2,[10,10,30,30]),c(1,[10,10,30,30])],100,100);assert r['selected_index']==1;cases.append('exact_equal_final_index_tie')
 for f in (51,52):
  r=t.step(f,[c(0,[70,70,90,90])],100,100);assert r['branch']=='held_bbox' and r['last_observed_frame']==50 and r['true_gap']==f-50
 r=t.step(53,[c(0,[70,70,90,90])],100,100);assert r['branch']=='reinitialized_identity_uncertain' and r['track_segment']==1;cases.append('two_holds_then_third_reinitialises_even_single_candidate')
 r=t.step(55,[],100,100);assert r['source_discontinuity'] and r['source_segment']==1 and r['last_observed_bbox'] is None;cases.append('source_discontinuity_clears')
 t=Tracker();t.step(0,[c(0,[0,0,13,10])],100,100);r=t.step(1,[c(0,[7,0,20,10])],100,100);assert iou([0,0,13,10],[7,0,20,10])==.3 and r['branch']=='associated_detection';cases.append('iou_exact_point30_accepted')
 assert not candidate(0,[0,0,float('nan'),10],.9,0,100,100)['valid'] and not c(0,[0,0,0,5])['valid'];cases.append('nonfinite_zero_area_rejected')
 from vendor.preprocess import _preprocess
 from vendor.letterbox import letterbox,unletterbox
 from vendor.nanodet import NanoDet
 crop,sx,sy,ss,cliff=_preprocess(np.zeros((100,200,3),np.uint8),np.array([0,0,200,100],np.float32),100,200)
 assert crop.shape==(3,224,224) and crop.dtype==np.float32 and np.array_equal(cliff,np.array([0,0,1],np.float32));cases.append('exact_extracted_preprocess_synthetic_no_model')
 sig=str(inspect.signature(NanoDet.__init__));assert 'modelPath' in sig and 'prob_threshold' in sig
 im,sc=letterbox(np.zeros((100,200,3),np.uint8));assert im.shape==(416,416,3) and unletterbox(np.array([0.,104.,416.,312.]),(100,200),sc).tolist()==[0,0,200,100];cases.append('paired_nano_function_binding_synthetic')
 assert nearest(2,[1,3])==1;cases.append('nearest_equal_distance_earlier')
 geoms={0:{'c':np.zeros(3),'x':np.array([1.,0,0]),'R':None,'d':1.,'b':.2},1:{'c':np.full(3,np.nan),'x':None,'R':None,'d':0.,'b':0.}}
 D,ds,pr=complete_proxy(geoms);assert ds==[0] and pr[0]['rotation_source']=='camera_axis_prior' and pr[1]['rotation_source']=='identity_orientation_prior' and pr[1]['translation_parent']==0;cases.append('fallback_camera_identity_and_held_translation')
 geoms[0]['R']=np.eye(3);D,ds,pr=complete_proxy(geoms);assert pr[1]['rotation_parent']==0;cases.append('fallback_nearest_formula_precedence')
 try:complete_proxy({0:{'c':np.zeros(3),'x':None,'R':None,'d':0.,'b':0.}})
 except ValueError as e:assert str(e)=='no_input_driven_mesh_dimensions'
 else:raise AssertionError('zero dimensions must fail')
 cases.append('no_dimensions_refused')
 cfg=load(Path(__file__).parent/'config_20261016.json');sys.path.insert(0,cfg['kit']);sys.path.insert(0,cfg['mesh_site'])
 from tools import pack_reconstruction as pack
 from v2dlb import mhr_submission as ms
 from v2dlb.mesh_common import _mesh_faces
 for n in [136,68,45]:assert np.array_equal(ms._unpack(pack.ms_pack(np.arange(n),n),n,str(n)),np.arange(n))
 bad=pack.ms_pack(np.arange(136),136);bad[-1,-1]=1
 try:ms._unpack(bad,136,'bad_tail')
 except ValueError:pass
 else:raise AssertionError('tail not rejected')
 for fun in [lambda:ms._rotations(np.array([np.diag([-1.,1,1])])),lambda:ms._object_scale(np.array([[0.,0,0]])),lambda:_mesh_faces(np.array([[0,1,8]]),8)]:
  try:fun()
  except ValueError:pass
  else:raise AssertionError('bad structural case not rejected')
 cases.append('official_pack_unpack_tail_R_scale_face_negative_cases')
 v,f=box_mesh(np.array([1.,.2,.2]));edges={}
 for row in f:
  for i in range(3):
   e=tuple(sorted([int(row[i]),int(row[(i+1)%3])]))
   edges[e]=edges.get(e,0)+1
 volume=sum(np.dot(a,np.cross(b,c))/6 for a,b,c in v[f]);assert len(edges)==18 and all(x==2 for x in edges.values()) and volume>0
 cases.append('closed_oriented_box')
 semantic=ast.parse(Path(cfg['assets']['semantics']['path']).read_text());scope={};exec(compile(semantic,'public_semantics','exec'),scope)
 expected={7:'left_elbow',8:'right_elbow',62:'left_wrist',41:'right_wrist',49:'left_forefinger_third_joint',61:'left_pinky_finger_third_joint',28:'right_forefinger_third_joint',40:'right_pinky_finger_third_joint'}
 assert all(scope['pose_info']['keypoint_info'][i]['name']==n for i,n in expected.items());cases.append('public_head70_semantic_binding')
 dump(out,{'date':'2026-10-16','status':'passed','synthetic':True,'cases':cases,'model_loads':0,'forwards':0,'nano_constructor_signature':sig,'preprocess_signature':str(inspect.signature(_preprocess)),'source_sha256':sha(__file__)})
 print(json.dumps({'status':'passed','cases':len(cases),'output':str(out)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)
