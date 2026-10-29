"""S20T-T01 fixed, nonrecursive original-window statistic; no IO or models."""
import math
import statistics
from fractions import Fraction

POLICY = dict(schema='r020_T1_stageA_policy_v1', half_window=3, window=7,
              mad_factor=1.4826, threshold=3.0, scale_epsilon_m=1e-6,
              cadence_num=1, cadence_den=30, center_must_be_only_outlier=True,
              strict_neighbor_extremum=True, median_inside_neighbor_interval=True,
              endpoint='inherit_no_padding', input='original_R0_only',
              proposal='cam_x_y_unchanged_target_z_median', delta_clipping=False,
              recursive=False, model_loads=0, forwards=0, rig=0, detector=0,
              video_decode=0, tuning_after_audit=False)

def context_reason(rows):
    if len(rows) != 7 or any(r is None for r in rows):
        return 'missing_context'
    for k in ('episode','source','segment_id','input_version','W','H'):
        if len({r[k] for r in rows}) != 1:
            return 'context_boundary'
    if not all(r['required'] for r in rows):
        return 'not_required'
    if not all(r['current'] for r in rows):
        return 'borrowed_or_unavailable'
    if not all(r['parent_closed'] for r in rows):
        return 'missing_parent_reference'
    if any(b['frame'] != a['frame']+1 for a,b in zip(rows,rows[1:])):
        return 'frame_gap'
    times = [Fraction(r['pts_num']*r['time_base_num'],
                      r['pts_den']*r['time_base_den']) for r in rows]
    if any(b-a != Fraction(1,30) for a,b in zip(times,times[1:])):
        return 'non30Hz'
    return None

def propose(z):
    if len(z) != 7 or not all(type(x) in (int,float) and math.isfinite(x) for x in z):
        return dict(reason='invalid_cam', proposed=False)
    m = statistics.median(z)
    scale = POLICY['mad_factor']*statistics.median(abs(x-m) for x in z)
    d = dict(median_m=m, scale_m=scale, proposed=False)
    if scale <= POLICY['scale_epsilon_m']:
        return dict(d, reason='scale_unresolved')
    outliers = [i for i,x in enumerate(z) if abs(x-m)>POLICY['threshold']*scale]
    if outliers != [3]:
        return dict(d, reason='cluster_or_no_impulse')
    lo,hi = sorted((z[2],z[4]))
    if not ((z[3]<lo or z[3]>hi) and lo<=m<=hi):
        return dict(d, reason='trend_or_turning_ambiguous')
    return dict(d, proposed=True, reason='statistical_proposal', target_z_m=m, delta_z_m=m-z[3])

def audit_segment(rows, cams):
    """Every decision uses the immutable input map; no target feeds another window."""
    result=[]
    for i,r in enumerate(rows):
        if i<3 or i>=len(rows)-3:
            result.append(dict(frame=r['frame'],reason='segment_endpoint',proposed=False,context_valid=False))
            continue
        window=rows[i-3:i+4]
        reason=context_reason(window)
        if reason:
            result.append(dict(frame=r['frame'],reason=reason,proposed=False,context_valid=False))
        else:
            d=propose([cams[x['frame']][2] for x in window])
            result.append(dict(d,frame=r['frame'],context_valid=True))
    return result
