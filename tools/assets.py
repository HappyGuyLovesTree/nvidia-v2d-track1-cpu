#!/usr/bin/env python3
"""Command entry; no work runs on import."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2d_public.assets import main
if __name__=="__main__":
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps({"status":"failed","error_type":type(exc).__name__,"message":str(exc)}),file=sys.stderr)
        raise SystemExit(1)
