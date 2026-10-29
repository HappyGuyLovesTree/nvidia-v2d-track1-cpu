"""CD-H-only production binding. Unknown outcomes retain the exclusive intent."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess
from .core import *
from v2d_public.inputs import SAMPLE_SHA, sample_rows
from v2d_public.packing import validate_candidate

SLUG='v2d-challenge-track1-cd-h'
UPLOADER_SHA='ede0c82183f940c8b2c81ae138169d61c1c626163d4a115362a9b30b9ff05461'
GATES=('full_coverage','provenance','invariants','rig','roundtrip','decoded_changed','quality','resources')
BOUND=('run_id','policy_sha256','source_tree_sha256','source_commit','commit_url')

def fresh(record):
    t=datetime.fromisoformat(record['observed_at'])
    need(t.tzinfo is not None and 0<=(datetime.now(timezone.utc)-t).total_seconds()<=300, 'platform observation not fresh within five minutes')

def evidence(c):
    need(c['schema']=='t1_cdh_contract_v1' and c['scope']=='official' and c['stage']=='cdh_send_authorized', 'production send stage required')
    verify_freeze(c['source_freeze'],c['source_tree_sha256'])
    git_identity(Path(__file__).resolve().parents[1],c['source_commit'])
    need(c['commit_url']=='https://github.com/HappyGuyLovesTree/nvidia-v2d-track1-cpu/commit/'+c['source_commit'], 'send source URL')
    full=json.loads(verify(c['full_result']).read_text())
    need(full['schema']=='t1_full_result_v1' and full['scope']=='official' and full['status']=='full_package_awaiting_independent_acceptance', 'fixture cannot enter send')
    need(full['candidate']==c['candidate'] and full['sample']==c['sample'], 'full candidate/sample binding')
    need(full['coverage']==dict(episodes=30,frames=9877,source_runs=65,rows=740780), 'full production coverage')
    need(full['accepted_frames']>0 and full['dense_changed_frames']>0 and full['decoded_changed_values']>0 and full['model_calls']==0,'actual changed values required')
    from .package import ORIGINAL32
    need(full['validation_protocol']=='prospective_all_A_proposals' and full['original_sets']==ORIGINAL32,'full validation protocol; old32 cannot become pass')
    for k in BOUND: need(full[k]==c[k], 'full binding '+k)
    root=Path(c['full_result']['path']).parent.parent
    need(c['full_supervisor_receipt']['path']==str(root/'supervisor_receipt.json'), 'supervisor receipt location')
    sup=json.loads(verify(c['full_supervisor_receipt']).read_text()); origin=json.loads(verify(c['full_origin']).read_text())
    need(c['full_origin']['path']==str(root/'origin.json') and origin['authority']['contract_sha256']==digest(json.loads(verify(full['contract']).read_text())), 'full supervisor authority')
    need(sup['status']=='completed' and sup['exit']==0 and sup['reap'] is not None and sup['run_id']==origin['run_id']==full['execution_run_id'] and origin['fixture'] is False, 'full supervisor completion/reap')
    ind=json.loads(verify(c['independent_receipt']).read_text())
    need(ind['schema']=='t1_independent_package_acceptance_v1' and ind['status']=='passed' and ind['scope']=='official' and ind['candidate']==c['candidate'] and ind['full_result']==c['full_result'], 'independent actual artifact binding')
    for k in BOUND: need(ind[k]==c[k], 'independent producer binding '+k)
    need(ind['validation_design']==full['validation_design'] and ind['validation_protocol']==full['validation_protocol'] and ind['original_sets']==ORIGINAL32,'independent full validation design')
    for gate in GATES:
        ev=json.loads(verify(ind['evidence'][gate]).read_text())
        need(ev['status']=='passed' and ev['candidate']==c['candidate'], 'missing bound independent evidence '+gate)
        for k in BOUND: need(ev[k]==c[k], 'independent evidence source mismatch')
    pub=json.loads(verify(c['public_code_evidence']).read_text())
    need(pub['accessible'] is True and pub['remote_commit']==c['source_commit'], 'same public commit unavailable')
    for k in ('source_tree_sha256','source_commit','commit_url','policy_sha256'): need(pub[k]==c[k], 'public code mapping')
    platform=json.loads(verify(c['platform_evidence']).read_text()); fresh(platform)
    need(platform['slug']==SLUG and platform['account']=='hanboliamli' and type(platform['team_id']) is int and platform['team_id']>0, 'target/account/current team unverified')
    need(platform['entered'] is True and platform['open'] is True and platform['daily_available']>=1 and platform['weekly_available']>=1, 'entry/open/allowance unverified')
    for name in ('identity','daily','weekly'): verify(platform['evidence'][name])
    recent=json.loads(verify(c['latest_submissions']).read_text()); fresh(recent)
    need(recent['slug']==SLUG and recent['account']==platform['account'] and recent['team_id']==platform['team_id'], 'dedup current team')
    for record in recent['records']:
        need(record.get('run_id')!=c['run_id'] and record.get('sha256')!=c['candidate']['sha256'], 'same run/hash: query only, never resend')
    need(c['uploader']['sha256']==UPLOADER_SHA and verify(c['uploader']), 'fixed uploader differs')
    need(sha(verify(c['sample']))==SAMPLE_SHA,'fixed official sample')
    validate_candidate(verify(c['candidate']),sample_rows(c['sample']['path']),c['commit_url'])
    return dict(full=full,platform=platform)

def final_argv(c,out):
    out=Path(out)
    return [c['uploader']['path'],str(out/'candidate.parquet'),'--track','track_1','--only','cd_h_cm','-m',c['run_id']+' sha256:'+c['candidate']['sha256'][:12],'--receipt',str(out/'official_receipt.json'),'--wait']

def bind(c,out):
    evidence(c); out=Path(out)
    need(out.is_absolute() and out.resolve()==out and not out.exists(), 'exclusive canonical binding output')
    out.mkdir(parents=True); snap=write_new(out/'candidate.parquet',verify(c['candidate']).read_bytes(),True)
    need(snap['sha256']==c['candidate']['sha256'] and snap['bytes']==c['candidate']['bytes'], 'snapshot changed')
    bound=dict(schema='t1_cdh_binding_v1',scope='official',validation_mode='complete',contract_sha256=digest(c),candidate=c['candidate'],snapshot=snap,argv=final_argv(c,out),**{k:c[k] for k in BOUND})
    write_new(out/'binding.json',bound,True); return bound

def send(c,bound,out):
    evidence(c); out=Path(out); need(out.resolve()==out and out.is_absolute(), 'send output')
    need(json.loads((out/'binding.json').read_text())==bound and bound['contract_sha256']==digest(c), 'persisted binding differs')
    need(bound['scope']=='official' and bound['validation_mode']=='complete' and bound['candidate']==c['candidate'], 'fixture/validateFalse cannot send')
    for k in BOUND: need(bound[k]==c[k], 'send binding differs')
    snap=verify(bound['snapshot']); need(snap==out/'candidate.parquet' and bound['snapshot']['sha256']==c['candidate']['sha256'], 'snapshot identity')
    argv=final_argv(c,out); need(bound['argv']==argv, 'final argv differs')
    receipt=out/'official_receipt.json'; need(not receipt.exists() and not receipt.is_symlink(), 'receipt already exists')
    validate_candidate(snap,sample_rows(c['sample']['path']),c['commit_url'])
    write_new(out/'send_intent.json',dict(state='unknown_until_queried',candidate=bound['snapshot'],argv=argv,run_id=c['run_id'],known_rate_limit_behavior='official uploader retries explicit rate limits up to five attempts; unknown outcomes never retried by this gate'),True)
    # No outer retry. Intent survives spawn failure, exception, nonzero or timeout.
    cp=subprocess.run(argv,text=True,capture_output=True)
    write_new(out/'send_process.json',dict(exit=cp.returncode,stdout=cp.stdout,stderr=cp.stderr,remote_byte_hash='not independently verified'),True)
    verify(bound['snapshot']); need(cp.returncode==0, 'unknown sending state; query existing receipt/submissions only')
    return dict(receipt=str(receipt),remote_byte_hash='not independently verified')
