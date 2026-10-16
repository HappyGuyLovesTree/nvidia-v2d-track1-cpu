"""Original proxy fallback, scoped to one source with episode dimensions. No model calls."""
import numpy as np
from common import nearest

def complete_proxy_source(geoms, dimensions):
 goodR=[f for f,g in geoms.items() if g['R'] is not None];goodC=[f for f,g in geoms.items() if np.isfinite(g['c']).all()]
 dims=[f for f,g in geoms.items() if np.isfinite([g['d'],g['b']]).all() and min(g['d'],g['b'])>0]
 if not dims and dimensions is None:raise ValueError('no_input_driven_mesh_dimensions')
 D=np.asarray(dimensions) if dimensions is not None else np.array([np.median([geoms[f]['d'] for f in dims]),np.median([geoms[f]['b'] for f in dims]),np.median([geoms[f]['b'] for f in dims])])
 result={}
 for f,g in geoms.items():
  R=g['R'];c=g['c'];rp=f;cp=f;kind='formula'
  if R is None:
   if goodR:rp=nearest(f,goodR);R=geoms[rp]['R'];kind='nearest_formula_rotation'
   elif g['x'] is not None and np.isfinite(g['x']).all():
    x=g['x'];axis=np.eye(3)[int(np.argmin(np.abs(x)))];y=axis-x*np.dot(axis,x);y/=np.linalg.norm(y);R=np.column_stack([x,y,np.cross(x,y)]);kind='camera_axis_prior';rp=None
   else:R=np.eye(3);kind='identity_orientation_prior';rp=None
  if not np.isfinite(c).all():
   if not goodC:raise ValueError('no_finite_translation')
   cp=nearest(f,goodC);c=geoms[cp]['c']
  assert np.isfinite(R).all() and np.isfinite(c).all() and np.max(np.abs(R.T@R-np.eye(3)))<=1e-6 and abs(np.linalg.det(R)-1)<=1e-6
  result[f]={'R':R,'t':c,'rotation_source':kind,'rotation_parent':rp,'rotation_distance':None if rp is None else abs(rp-f),
   'translation_parent':cp,'translation_distance':abs(cp-f),'held_translation':cp!=f}
 return D,dims,result
