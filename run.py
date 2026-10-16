#!/usr/bin/env python3
"""Freeze provenance and launch each stage once, under independent measured budgets."""
from pathlib import Path
import argparse,datetime,importlib,importlib.metadata as md,json,os,subprocess,sys,time
from common import sha,dump,load
P=Path(__file__).resolve().parent

def git(*args):return subprocess.check_output(['git','-C',str(P),*args],text=True).strip()
def clean_commit():
 assert git('status','--porcelain')=='','source repository must be clean'
 return git('rev-parse','HEAD')
def size(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
def prepare(args,cfg):
 assert not (args.output/'run_manifest_20261016.json').exists(),'existing run manifest preserved'
 commit=clean_commit();assets={}
 for n,x in cfg['assets'].items():
  actual=sha(x['path']);assert actual==x['sha256'],n+' identity mismatch';assets[n]={**x,'actual_bytes':Path(x['path']).stat().st_size}
 assert args.stop>=args.start and args.stop-args.start+1<=cfg['limits']['A']['max_frames']
 assert (args.episode,args.start,args.stop)==(cfg['authorised_episode'],cfg['authorised_start'],cfg['authorised_stop']),'range outside this frozen authorisation'
 for d in ['source','detector','hmr','identity','rig','proxy','episodes','pack','logs','overlays']:(args.output/d).mkdir(parents=True,exist_ok=True)
 files={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
 kitfiles={str(p.relative_to(Path(cfg['kit']))):sha(p) for p in Path(cfg['kit']).rglob('*.py')}
 manifest={'task':'T010.04','date':'2026-10-16','started_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'repo':str(P),'commit':commit,'tree':git('rev-parse','HEAD^{tree}'),'repo_identity':{'name':git('config','user.name'),'email':git('config','user.email')},'remote_status':'unpublished_no_remote','output':str(args.output),'episode':args.episode,'start':args.start,'stop':args.stop,'assets':assets,'source_files':files,'kit_files':kitfiles,'config_sha256':sha(P/'config_20261016.json'),'entry':str(P/'run.py'),'python':sys.executable,'cwd':os.getcwd(),'spec':cfg['spec'],'identity_sha256':'pending_until_B','mesh_sha256':'pending_until_B','provenance_kind':'local_git','submission_eligible':False,'network_in_model_stages':False,'raw_vs_processed':'hmr/raw unchanged; identity/rig/proxy/episodes derived'}
 dump(args.output/'run_manifest_20261016.json',manifest);print(json.dumps({'prepared':True,'commit':commit,'tree':manifest['tree'],'manifest':str(args.output/'run_manifest_20261016.json')},indent=2))
def launch(args,cfg):
 import psutil
 man=load(args.output/'run_manifest_20261016.json');assert clean_commit()==man['commit']
 assert sha(P/'config_20261016.json')==man['config_sha256']
 for p,h in man['source_files'].items():assert sha(P/p)==h,p
 for prev in {'A':[],'B':['A'],'C':['A','B']}[args.stage]:assert load(args.output/'logs'/f'{prev}_receipt_20261016.json')['status']=='completed'
 lock=args.output/'logs'/f'{args.stage}_one_launch_20261016.lock'
 with lock.open('x') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
 budget=cfg['limits'][args.stage];cmd=[sys.executable,str(P/'worker.py'),'--stage',args.stage,'--output',str(args.output)]
 before=size(args.output);start=time.monotonic();peak=0;reason=None
 with (args.output/'logs'/f'{args.stage}_stdout_20261016.log').open('w') as stdout,(args.output/'logs'/f'{args.stage}_stderr_20261016.log').open('w') as stderr,(args.output/'logs'/f'{args.stage}_resources_20261016.jsonl').open('w') as monitor:
  env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'})
  proc=subprocess.Popen(cmd,cwd=P,env=env,stdout=stdout,stderr=stderr,start_new_session=True);pp=psutil.Process(proc.pid);lastdisk=before
  while proc.poll() is None:
   elapsed=time.monotonic()-start
   try:resident=pp.memory_info().rss+sum(c.memory_info().rss for c in pp.children(recursive=True))
   except psutil.Error:resident=0
   peak=max(peak,resident);lastdisk=size(args.output)
   monitor.write(json.dumps({'elapsed_s':elapsed,'rss_bytes':resident,'peak_rss_bytes':peak,'output_delta_bytes':lastdisk-before})+'\n');monitor.flush()
   if elapsed>budget['wall_s']:reason='wall_limit'
   elif resident>budget['rss_bytes']:reason='rss_limit'
   elif lastdisk-before>budget['output_bytes']:reason='output_limit'
   if reason:
    proc.terminate()
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    break
   time.sleep(.2)
  code=proc.wait()
 receipt={'stage':args.stage,'command':cmd,'cwd':str(P),'python':sys.executable,'commit':man['commit'],'returncode':code,'stop_reason':reason,'elapsed_wall_s':time.monotonic()-start,'sampled_peak_rss_bytes':peak,'output_delta_bytes':size(args.output)-before,'budget':budget,'source_clean_after':git('status','--porcelain')=='','source_commit_after':git('rev-parse','HEAD'),'network_allowed':False,'max_rss_sampling_interval_s':.2,'retries':0}
 dump(args.output/'logs'/f'{args.stage}_supervision_20261016.json',receipt);print(json.dumps(receipt,indent=2));sys.exit(0 if code==0 and reason is None else 1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','A','B','C'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--episode',type=int,default=16);ap.add_argument('--start',type=int,default=50);ap.add_argument('--stop',type=int,default=81);args=ap.parse_args();args.output=args.output.resolve();cfg=load(P/'config_20261016.json')
 (prepare if args.stage=='prepare' else launch)(args,cfg)
if __name__=='__main__':main()
