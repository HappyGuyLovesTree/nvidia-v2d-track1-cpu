#!/usr/bin/env python3
"""Portable CLI. No private paths or model execution permission."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2d_t1.core import write_new,source_freeze,digest,need,reference

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','plan','run','worker','recheck','send'])
    p.add_argument('--contract');p.add_argument('--output',required=True);p.add_argument('--supervision-root');p.add_argument('--worker-python');p.add_argument('--candidate');p.add_argument('--view');p.add_argument('--only',choices=['cd_h_cm'])
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]; out=Path(a.output)
    if a.action=='freeze': write_new(out,source_freeze(root),True);return
    if a.action=='send':
        from v2d_t1.gate import bind,send
        need(a.only=='cd_h_cm','explicit --only cd_h_cm required')
        c=json.loads(Path(a.contract).read_text()); b=bind(c,out);send(c,b,out);return
    from v2d_t1.package import contract_header,load_contract,worker,roundtrip,assemble
    if a.action=='worker':worker(a.contract,out,a.supervision_root);return
    if a.action=='run':
        from v2d_t1.supervisor import supervise
        c=contract_header(a.contract)
        need(out.is_absolute() and out.resolve()==out and not out.exists(),'new canonical supervisor root')
        need(a.worker_python==c['stock_runtime']['path'] and Path(a.worker_python).is_file(),'explicit pinned existing worker Python')
        out.mkdir(parents=True); cr=write_new(out/'contract.json',c,True)
        argv=[a.worker_python,*(['-O']*sys.flags.optimize),'-B',str(Path(__file__).resolve()),'worker','--contract',cr['path'],'--output',str(out/'work'),'--supervision-root',str(out)]
        result=supervise(out,argv,c['resource_policy'],authority=dict(contract_sha256=digest(c),stage=c['stage']),fixture=c['scope']!='official',initial_carry=c.get('initial_carry'))
        need(result['status']=='completed','full worker stopped; persistent receipt retained; no automatic retry');return
    if a.action=='plan':
        c=contract_header(a.contract)
        write_new(out,dict(status='header_and_execution_source_only',scope=c['scope'],contract=reference(a.contract),declared_episodes=len(c['episodes']),declared_frames=sum(len(x['parents']) for x in c['episodes']),retained_rig_arrays_read=False,dense_arrays_read=False,sample_or_D_validation=False,submission_ready=False),True);return
    need(contract_header(a.contract)['scope']=='fixture_only_never_send','standalone recheck is fixture-only; official independent review needs its own supervised card')
    c,r,rows,plan,d=load_contract(a.contract)
    if a.action=='recheck':
        need(c['scope']=='fixture_only_never_send','standalone recheck is fixture-only; official independent review needs its own supervised card')
        v=assemble(c,r,plan,d,out/'recomputed_view');checked=roundtrip(c,out/'recomputed_view',a.candidate)
        write_new(out/'recheck.json',dict(status='local_recheck_completed_not_independent_acceptance',scope=c['scope'],candidate=reference(a.candidate),provenance=reference(out/'recomputed_view/provenance.json'),roundtrip=checked),True)

if __name__=='__main__': main()
