"""Portable dependencies for the byte-identical geometry_d numerical bodies.

Dataset cardinalities and permissions live in the caller's manifest, not here.
The original production metadata policy hash must not be relabelled as this file.
"""
import hashlib
import numpy as np
from geometry import need,ContractError
from control import compact_digest

POLICY=dict(rig_scored22=[1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42])
GEOMETRY_POLICY=dict(schema='t1_portable_geometry_policy_v1',rig_scored22=POLICY['rig_scored22'],
 q_delta_factor_z=-10.,object_delta_factor_z=1.,rig_vertices=18439,rig_joints=127,rigid_translation_tolerance_m=1e-4,
 minimum_camera_Z_m=.05,max_projection_displacement_px=2.,computation_dtype='float64_then_original_storage_dtype',
 allowed_slots=['pose[2]','object_translation[2]'],object_transform='(vertices*scale)@R.T+t',
 rejection='whole_frame_exact_R0_copy',persistence='all native rig plus full raw forward',streaming=False)
MODEL_SHA='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
def bit_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def identity(a):return dict(shape=list(a.shape),dtype=a.dtype.str,bytes_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
def output_identity(arrays):return compact_digest({k:identity(v) for k,v in sorted(arrays.items())})
def camera(native):return {k:v.astype(np.float64)*np.asarray([1.,-1.,-1.])/100 for k,v in native.items()}
