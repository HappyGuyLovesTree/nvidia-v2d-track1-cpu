"""Behavior and frozen raw replay checks. No constructor/forward/image/video decoding."""
from pathlib import Path
from copy import deepcopy
import argparse, collections, json, time, resource
import numpy as np
from common import Tracker, candidate, sha
from tracking import LowScoreTracker, METHOD, RULES, digest, validate_raw

IDENTITY={k:'1'*64 for k in ('video_sha256','detector_sha256','postprocess_sha256','runtime_sha256')}
NUMERIC=digest(RULES)
PARENTS={'raw_sha256':'2'*64,'rgb_sha256':'3'*64,'blob_sha256':'4'*64}

def new():return LowScoreTracker(identity=IDENTITY,numeric_config_hash=NUMERIC)
def c(index,box=(0,0,30,10),score=.9):return candidate(index,box,score,0,100,100)
def step(t,fid,cs,**kw):
    return t.step(fid,cs,100,100,**dict(episode=kw.get('episode',16),source_segment=kw.get('source_segment',0),pts=kw.get('pts',fid),time_base='1/30',input_parents=kw.get('input_parents',PARENTS)))
def restore(cp,**kw):
    return LowScoreTracker.restore_state(cp,identity=kw.get('identity',IDENTITY),numeric_config_hash=kw.get('numeric_config_hash',NUMERIC),expected_state_sha256=kw.get('expected_state_sha256',cp.get('state_sha256')),expected_input_parents=kw.get('expected_input_parents',PARENTS),expected_context=kw.get('expected_context',{k:cp.get('state',{}).get(k) for k in ('episode','source_segment','previous_frame','previous_pts','time_base','width','height')}))
def refused(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('expected explicit rejection')

def synthetic():
    cases=[]
    # Threshold tests use exact rational geometry: [0,0,13,10] vs [7,0,20,10] -> .30;
    # [0,0,30,10] vs [10,0,40,10] -> .50. Perturb coordinates on each side.
    for score,expected in ((np.nextafter(.1,0),False),(.1,False),(np.nextafter(.1,1),True)):
        t=new();step(t,0,[c(0)]);r=step(t,1,[c(1,score=score)])
        assert (r['branch']=='associated_low_detection')==expected
    cases.append('score_floor_exact_and_sides')
    for score,tier in ((np.nextafter(.35,0),'low'),(.35,'high'),(np.nextafter(.35,1),'high')):
        t=new();step(t,0,[c(0)]);r=step(t,1,[c(1,score=score)]);assert r['selected_tier']==tier
        empty=new();r=step(empty,0,[c(0,score=score)]);assert (r['branch']=='initialized_detection')==(tier=='high')
    cases.append('high_threshold_exact_and_sides_low_never_initializes')
    for dx,ok in ((-1e-9,True),(0,True),(1e-9,False)):
        t=new();step(t,0,[c(0,(0,0,13,10))]);r=step(t,1,[c(1,(7+dx,0,20+dx,10))]);assert (r['branch']=='associated_detection')==ok
        t=new();step(t,0,[c(0)]);r=step(t,1,[c(1,(10+dx,0,40+dx,10),.2)]);assert (r['branch']=='associated_low_detection')==ok
    cases.append('both_iou_thresholds_exact_and_sides')
    t=new();step(t,0,[c(0)]);r=step(t,1,[c(2,(0,0,30,10),.34),c(1,(5,0,35,10),.35)])
    assert r['selected_index']==1 and r['selected_tier']=='high';cases.append('high_priority_over_better_low_iou')
    for candidates,expected in (([c(2),c(1)],1),([c(0,(5,0,35,10)),c(1)],1),([c(0,score=.8),c(1)],1),([c(0,(0,0,20,10)),c(1)],1)):
        r=step(new(),0,candidates);assert r['selected_index']==expected
    t=new();step(t,0,[c(0)]);r=step(t,1,[c(2),c(1)]);assert r['selected_index']==1
    cases.append('stable_init_and_association_ties')
    for with_high in (False,True):
        t=new();step(t,0,[c(0)])
        for f in (1,2):
            r=step(t,f,[]);assert r['branch']=='held_bbox' and r['score'] is None and r['last_observed_frame']==0 and r['true_gap']==f
        r=step(t,3,[c(1,(70,70,90,90),.9 if with_high else .2)])
        assert r['branch']==('reinitialized_identity_uncertain' if with_high else 'full_no_detection')
        if not with_high:assert t.state['last_high_frame'] is None and t.state['last_bbox'] is None
    cases.append('two_holds_third_restart_or_clear')
    t=new();step(t,0,[c(0)])
    for f in range(1,20):
        r=step(t,f,[c(1,score=.2)]);assert r['last_observed_frame']==f and r['lost_count']==0 and r['last_high_frame']==0 and r['consecutive_low_matches']==f
    step(t,20,[c(0)]);assert t.state['consecutive_low_matches']==0 and t.state['last_high_frame']==20
    cases.append('long_low_chain_no_hidden_cap_and_high_audit_reset')
    for kwargs,fid in (({},2),({'source_segment':1},1),({'episode':17},1)):
        t=new();step(t,0,[c(0)]);r=step(t,fid,[c(1,score=.2)],**kwargs)
        assert r['source_discontinuity'] and r['branch']=='full_no_detection' and r['last_high_frame'] is None
    cases.append('gap_source_episode_clear_low_cannot_bridge')
    t=new();step(t,0,[c(0)])
    before=t.export_state();refused(lambda:step(t,1,[],pts=0));assert t.export_state()==before
    refused(lambda:t.step(1,[],101,100,episode=16,source_segment=0,pts=1,time_base='1/30',input_parents=PARENTS))
    cases.append('pts_dimensions_errors_do_not_mutate_state')
    stream=[[c(0)],[c(1,score=.2)],[],[],[c(2,(70,70,90,90))],[c(3,(70,70,90,90),.2)],[]]
    t=new();expected=[step(t,f,cs) for f,cs in enumerate(stream)];final=t.export_state()
    for cut in range(1,len(stream)):
        a=new();actual=[step(a,f,cs) for f,cs in enumerate(stream[:cut])]
        a=restore(json.loads(json.dumps(a.export_state())))
        actual += [step(a,f,stream[f]) for f in range(cut,len(stream))]
        assert actual==expected and a.export_state()==final
    cases.append('all_synthetic_breakpoints_decision_state_equal')
    cp=t.export_state()
    refused(lambda:restore({'state':{'last':[0,0,1,1]}}))
    for field,value in (('method','single_iou_hold_v1'),('schema','old'),('numeric_config_hash','f'*64),('identity',{**IDENTITY,'video_sha256':'f'*64})):
        bad=deepcopy(cp);bad['state'][field]=value;bad['state_sha256']=digest(bad['state']);refused(lambda:restore(bad))
    context={k:cp['state'][k] for k in ('episode','source_segment','previous_frame','previous_pts','time_base','width','height')}
    for key,value in (('episode',17),('source_segment',1),('previous_frame',99),('previous_pts',99),('time_base','1/60'),('width',200),('height',200)):
        refused(lambda:restore(cp,expected_context={**context,key:value}))
    refused(lambda:restore(cp,expected_input_parents={'raw_sha256':'f'*64}))
    refused(lambda:restore(cp,expected_state_sha256='f'*64))
    bad=deepcopy(cp);bad['state']['lost_count']=99;refused(lambda:restore(bad))
    cases.append('legacy_method_schema_numeric_identity_parent_and_corruption_refused')
    for box in ([0,0,1],[0,0,float('nan'),10],[0,0,0,10],[20,0,10,10]):
        t=new();bad={'index':0,'xyxy':box,'score':.9,'class_id':0,'valid':True};refused(lambda:step(t,0,[bad]))
    refused(lambda:step(new(),0,[{**c(0),'score':float('nan')}]))
    refused(lambda:step(new(),0,[c(0),c(0)]))
    t=new();r=step(t,0,[c(0,(-5,-5,20,20))]);assert r['bbox']==[0.,0.,20.,20.]
    r=step(new(),0,[{**c(0),'valid':False,'xyxy':None,'score':None}]);assert r['branch']=='full_no_detection'
    cases.append('malformed_nonfinite_degenerate_duplicate_rejected_clipping_invalid_audit')
    from tracking import RAW_SHAPES
    raw=[np.zeros(s,np.float32) for s in RAW_SHAPES];validate_raw(raw)
    refused(lambda:validate_raw(raw[:4]));refused(lambda:validate_raw({f'output_{i}':v for i,v in enumerate(raw+[raw[0]])}))
    bad=raw.copy();bad[0]=bad[0].astype(np.float64);refused(lambda:validate_raw(bad))
    bad=raw.copy();bad[0]=np.zeros((1,1,80),np.float32);refused(lambda:validate_raw(bad))
    bad=raw.copy();bad[1]=bad[1].copy();bad[1][0,0,0]=np.nan;refused(lambda:validate_raw(bad))
    cases.append('six_raw_keys_shapes_dtype_finite')
    return cases

def replay(root):
    from vendor.nanodet import NanoDet
    from vendor.letterbox import unletterbox
    frozen=json.loads((root/'DD/证据/Q011.02/offline_postprocess_20261017.json').read_text())
    exp=root/'DD/实验/T010.04_连续32'
    for name in ('common.py','vendor/nanodet.py','vendor/letterbox.py'):
        p=Path(__file__).parent/name
        assert sha(p)==frozen['source_hashes'][str(p)]
    d=NanoDet.__new__(NanoDet)  # metadata only, never __init__/readNet
    d.strides=(8,16,32,64);d.image_shape=(416,416);d.reg_max=7;d.project=np.arange(8);d.iou_threshold=.6;d.anchors_mlvl=[]
    for stride in d.strides:
        n=416//stride;x,y=np.meshgrid(np.arange(n)*stride,np.arange(n)*stride)
        d.anchors_mlvl.append(np.column_stack((x.ravel()+.5*(stride-1),y.ravel()+.5*(stride-1))))
    baseline=Tracker();low=new();rows=[];cache=[]
    for fid in range(50,82):
        p=exp/'detector'/f'raw_{fid:06d}.npz';q=exp/'detector'/f'decision_{fid:06d}.json'
        assert sha(p)==frozen['parents'][str(fid)]['raw'] and sha(q)==frozen['parents'][str(fid)]['decision']
        old=json.loads(q.read_text());identity=old['identity'];w=identity['width'];h=identity['height']
        with np.load(p,allow_pickle=False) as z:raw=validate_raw(dict(z))
        def decode(threshold):
            d.prob_threshold=threshold;pred=d.post_process(raw)
            return [candidate(i,unletterbox(r[:4],(h,w),old['letterbox_scale']),float(r[4]),int(r[5]),w,h) for i,r in enumerate(pred)] if pred.size else []
        b=baseline.step(fid,decode(.35),w,h)
        assert b['branch']==old['branch'] and b['bbox']==old['bbox'] and b['selected_index']==old['selected_index']
        cs=decode(.1)
        ctx=dict(episode=16,source_segment=0,pts=identity['pts'],time_base=identity['time_base'],input_parents={'raw_sha256':sha(p),'rgb_sha256':identity['rgb_sha256']})
        r=low.step(fid,cs,w,h,**ctx);reference=frozen['rows'][fid-50]
        assert r['branch']==reference['candidate_branch'] and r['bbox']==reference['candidate_bbox'] and r['selected_index']==reference['selected_index']
        rows.append({'frame':fid,'baseline_branch':b['branch'],'branch':r['branch'],'bbox':r['bbox'],'changed_bbox':r['bbox']!=b['bbox'],'decision_sha256':digest(r),'state_sha256':low.export_state()['state_sha256']})
        cache.append((fid,cs,w,h,ctx,r))
    for cut in (1,3,13,20,31):
        t=new()
        for args in cache[:cut]:t.step(*args[:4],**args[4])
        cp=t.export_state();t=restore(cp,expected_input_parents=cache[cut-1][4]['input_parents'])
        for args in cache[cut:]:assert t.step(*args[:4],**args[4])==args[5]
        assert t.export_state()==low.export_state()
    changed=[r['frame'] for r in rows if r['changed_bbox']]
    assert changed==[51,52,53,*range(60,72),73]
    assert dict(collections.Counter(r['branch'] for r in rows))=={'initialized_detection':1,'associated_low_detection':16,'associated_detection':15}
    return {'baseline_branches':dict(collections.Counter(r['baseline_branch'] for r in rows)),'candidate_branches':dict(collections.Counter(r['branch'] for r in rows)), 'changed_bbox_frames':changed,'restore_breakpoints':[51,53,63,70,81],'rows':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    start=time.perf_counter();cases=synthetic();result=replay(args.root)
    result.update(schema='tracking_local_checks_v1',date='2026-10-17',status='passed',synthetic_cases=cases,synthetic_case_count=len(cases),new_model_loads=0,new_forwards=0,image_decodes=0,video_decodes=0,runtime={'numpy':np.__version__,'python':__import__('sys').version,'executable':__import__('sys').executable,'cv2':__import__('cv2').__version__},elapsed_s=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_hashes={n:sha(Path(__file__).parent/n) for n in ('tracking.py','check_tracking.py','worker.py','common.py','vendor/nanodet.py','vendor/letterbox.py')})
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
