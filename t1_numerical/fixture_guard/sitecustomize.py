"""Optional fixture-only import firewall, enabled by the selftest subprocesses."""
import os,sys
if os.environ.get('T1_FIXTURE_IMPORT_GUARD')=='1':
    def hook(kind,args):
        if kind=='import' and args[0].split('.')[0] in ('torch','onnxruntime','cv2','av','transformers'):
            raise RuntimeError('fixture forbids real model/video runtime import')
    sys.addaudithook(hook)
