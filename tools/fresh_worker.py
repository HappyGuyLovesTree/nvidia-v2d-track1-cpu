#!/usr/bin/env python3
"""Lease-checked fresh child. Direct invocation cannot enter production resume."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2d_public.fresh_worker import main
if __name__=='__main__':raise SystemExit(main())
