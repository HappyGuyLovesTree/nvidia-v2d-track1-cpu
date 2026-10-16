"""Deterministic single-target association. No model, filesystem, or runtime imports.

Caller owns input hashes and atomic persistence. See TRACKING_API_20261017.md.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import numpy as np

METHOD = 'single_iou_low_v1'
SCHEMA = 'single_target_tracker_state_v1'
RULES = {'score_floor': .10, 'high_score': .35, 'high_iou': .30,
         'low_iou': .50, 'nms_iou': .60, 'hold_frames': 2,
         'sort': 'float64:-iou,-score,-area,x1,y1,x2,y2,index'}
RAW_SHAPES = ((1,2704,80),(1,2704,32),(1,676,80),(1,676,32),(1,169,80),(1,169,32))
IDENTITY_KEYS = {'video_sha256','detector_sha256','postprocess_sha256','runtime_sha256'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

RULES_HASH = digest(RULES)

def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def _identity(identity, numeric_config_hash):
    if not isinstance(identity, dict) or not IDENTITY_KEYS <= identity.keys():
        raise ValueError('missing tracker input identity')
    if not all(_hash(identity[k]) for k in IDENTITY_KEYS) or not _hash(numeric_config_hash):
        raise ValueError('invalid identity/config hash')
    digest(identity)

def validate_raw(raw):
    """Require output_0..5 exactly, or an ordered six-array sequence; return arrays."""
    if isinstance(raw, dict):
        if set(raw) != {f'output_{i}' for i in range(6)}:
            raise ValueError('detector raw keys must be output_0..5')
        raw = [raw[f'output_{i}'] for i in range(6)]
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        raise ValueError('detector requires six paired tensors')
    for i, (a, shape) in enumerate(zip(raw, RAW_SHAPES)):
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'invalid detector output_{i}: expected finite float32 {shape}')
    return raw

def _box(box, w, h):
    a = np.asarray(box, dtype=np.float64)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError('bbox must be finite xyxy[4]')
    a = np.clip(a, [0,0,0,0], [w,h,w,h])
    if a[2] <= a[0] or a[3] <= a[1]:
        raise ValueError('bbox has nonpositive clipped area')
    return a.tolist()

def iou(a, b):
    a, b = np.asarray(a,np.float64), np.asarray(b,np.float64)
    wh = np.maximum(0, np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter = float(np.prod(wh))
    return inter / float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)

def _initial(c):
    return [-c['score'], -c['area'], *c['xyxy'], c['index']]

def _time(pts, time_base):
    if type(pts) is not int:
        raise ValueError('PTS must be integer')
    try:
        tb = Fraction(time_base)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        raise ValueError('invalid time_base') from e
    if tb <= 0:
        raise ValueError('time_base must be positive')
    return pts * tb

class LowScoreTracker:
    def __init__(self, *, identity, numeric_config_hash):
        _identity(identity, numeric_config_hash)
        self.state = dict(schema=SCHEMA, method=METHOD, numeric_config_hash=numeric_config_hash,
            rules_hash=RULES_HASH, identity=deepcopy(identity), episode=None, source_segment=None,
            previous_frame=None, previous_pts=None, time_base=None, width=None, height=None,
            last_bbox=None, last_observed_frame=None, lost_count=0, track_segment=-1,
            last_high_frame=None, consecutive_low_matches=0, last_input_parents=None)

    def export_state(self):
        s = deepcopy(self.state)
        return {'state': s, 'state_sha256': digest(s)}

    @classmethod
    def restore_state(cls, checkpoint, *, identity, numeric_config_hash, expected_state_sha256, expected_input_parents, expected_context):
        obj = cls(identity=identity, numeric_config_hash=numeric_config_hash)
        if not isinstance(checkpoint, dict) or set(checkpoint) != {'state','state_sha256'}:
            raise ValueError('invalid checkpoint envelope / legacy state rejected')
        s = deepcopy(checkpoint['state'])
        if not isinstance(s, dict) or set(s) != set(obj.state):
            raise ValueError('invalid state schema')
        if digest(s) != checkpoint['state_sha256'] or checkpoint['state_sha256'] != expected_state_sha256:
            raise ValueError('state hash mismatch')
        for k in ('schema','method','numeric_config_hash','rules_hash','identity'):
            if s[k] != obj.state[k]:
                raise ValueError(f'restore identity mismatch: {k}')
        context_keys = {'episode','source_segment','previous_frame','previous_pts','time_base','width','height'}
        if not isinstance(expected_context,dict) or set(expected_context) != context_keys or any(s[k] != expected_context[k] for k in context_keys):
            raise ValueError('restore context mismatch')
        if s['last_input_parents'] != expected_input_parents:
            raise ValueError('restore frame parent mismatch')
        for k in ('lost_count','consecutive_low_matches','track_segment'):
            if type(s[k]) is not int or s[k] < (-1 if k == 'track_segment' else 0):
                raise ValueError('invalid tracker counters')
        if s['lost_count'] > 2:
            raise ValueError('invalid loss state')
        if s['previous_frame'] is not None:
            if type(s['previous_frame']) is not int or s['previous_frame'] < 0 or s['episode'] is None or s['source_segment'] is None:
                raise ValueError('invalid frame identity')
            if type(s['width']) is not int or type(s['height']) is not int or min(s['width'],s['height']) <= 0:
                raise ValueError('invalid state dimensions')
            _time(s['previous_pts'],s['time_base'])
            if not isinstance(s['last_input_parents'],dict) or not s['last_input_parents'] or not all(_hash(v) for v in s['last_input_parents'].values()):
                raise ValueError('invalid state parents')
        if s['previous_frame'] is None and s != obj.state:
            raise ValueError('invalid initial state')
        if s['last_bbox'] is None:
            if s['last_observed_frame'] is not None or s['lost_count'] or s['last_high_frame'] is not None or s['consecutive_low_matches']:
                raise ValueError('inconsistent empty state')
        else:
            if _box(s['last_bbox'],s['width'],s['height']) != s['last_bbox']:
                raise ValueError('state bbox outside image')
            if not (type(s['last_observed_frame']) is int and type(s['last_high_frame']) is int and s['last_high_frame'] <= s['last_observed_frame'] <= s['previous_frame'] and s['track_segment'] >= 0):
                raise ValueError('inconsistent observation state')
            if s['previous_frame']-s['last_observed_frame'] != s['lost_count'] or s['consecutive_low_matches'] > s['last_observed_frame']-s['last_high_frame']:
                raise ValueError('inconsistent loss/audit history')
        obj.state = s
        return obj

    def step(self, fid, candidates, w, h, *, episode, source_segment, pts, time_base, input_parents):
        if type(fid) is not int or fid < 0 or type(w) is not int or type(h) is not int or min(w,h) <= 0:
            raise ValueError('invalid frame/dimensions')
        if type(episode) not in (str,int) or type(source_segment) not in (str,int):
            raise ValueError('missing episode/source identity')
        if not isinstance(input_parents,dict) or not input_parents or not all(_hash(v) for v in input_parents.values()):
            raise ValueError('missing frame parent hashes')
        now = _time(pts,time_base)
        s = deepcopy(self.state); before = digest(s)
        same_episode = s['episode'] == episode
        same_source = same_episode and s['source_segment'] == source_segment
        if s['previous_frame'] is not None and same_episode:
            if (w,h) != (s['width'],s['height']):
                raise ValueError('dimensions changed within episode')
            if now <= _time(s['previous_pts'],s['time_base']):
                raise ValueError('nonincreasing PTS')
        discontinuity = s['previous_frame'] is not None and (not same_source or fid != s['previous_frame']+1)
        if s['previous_frame'] is None or discontinuity:
            s.update(last_bbox=None,last_observed_frame=None,lost_count=0,last_high_frame=None,consecutive_low_matches=0)
            if not same_episode:
                s['track_segment'] = -1
        s.update(episode=episode,source_segment=source_segment,previous_frame=fid,
                 previous_pts=pts,time_base=str(Fraction(time_base)),width=w,height=h,last_input_parents=deepcopy(input_parents))
        cs = deepcopy(candidates)
        if not isinstance(cs,list):
            raise ValueError('candidates must be list')
        seen = set(); valid = []
        for c in cs:
            if not isinstance(c,dict) or type(c.get('index')) is not int or c['index'] < 0 or c['index'] in seen:
                raise ValueError('invalid/duplicate candidate index')
            seen.add(c['index'])
            # Explicit invalid candidate records remain auditable, never enter sorting.
            if not c.get('valid',True):
                c.update(tier='rejected',iou=None,initial_key=None,association_key=None)
                continue
            box = _box(c.get('xyxy'),w,h)
            score = c.get('score')
            if isinstance(score,bool) or not isinstance(score,(int,float,np.number)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('invalid score')
            if type(c.get('class_id')) is not int:
                raise ValueError('invalid class_id')
            score = float(score)
            tier = 'high' if score >= .35 else ('low' if score > .10 else 'excluded')
            c.update(xyxy=box,score=score,area=float((box[2]-box[0])*(box[3]-box[1])),tier=tier)
            c['initial_key'] = _initial(c)
            c['iou'] = iou(box,s['last_bbox']) if s['last_bbox'] is not None else None
            c['association_key'] = [-c['iou'],*c['initial_key']] if c['iou'] is not None else None
            if c['class_id'] == 0 and tier != 'excluded':
                valid.append(c)
        high = [c for c in valid if c['tier']=='high']; low = [c for c in valid if c['tier']=='low']
        chosen = None; branch = 'full_no_detection'; reason = 'no_valid_high_person'
        bbox = [0,0,w,h]; score = None
        if s['last_bbox'] is not None:
            for pool, threshold, label in ((high,.30,'associated_detection'),(low,.50,'associated_low_detection')):
                matches = [c for c in pool if c['iou'] >= threshold]
                if matches:
                    chosen = min(matches,key=lambda c:c['association_key']); branch=label; reason='high_iou_match' if pool is high else 'low_iou_match'
                    break
            if chosen is None:
                s['lost_count'] += 1
                if s['lost_count'] <= 2:
                    branch='held_bbox'; bbox=s['last_bbox'].copy(); reason='no_iou_match'
                else:
                    s.update(last_bbox=None,last_observed_frame=None,lost_count=0,last_high_frame=None,consecutive_low_matches=0)
                    reason='third_loss_cleared_state'
                    if high:
                        chosen=min(high,key=_initial);branch='reinitialized_identity_uncertain';reason='third_loss_high_restart'
        elif high:
            chosen=min(high,key=_initial);branch='initialized_detection';reason='high_initialization'
        if chosen is not None:
            if branch in ('initialized_detection','reinitialized_identity_uncertain'):
                s['track_segment'] += 1
            s.update(last_bbox=chosen['xyxy'].copy(),last_observed_frame=fid,lost_count=0)
            if chosen['tier']=='high':
                s.update(last_high_frame=fid,consecutive_low_matches=0)
            else:
                s['consecutive_low_matches'] += 1
            bbox=s['last_bbox'].copy();score=chosen['score']
        decision=dict(frame_index=fid,episode=episode,source_segment=source_segment,
            source_discontinuity=discontinuity,track_segment=s['track_segment'],method=METHOD,
            branch=branch,bbox=bbox,score=score,selected_index=None if chosen is None else chosen['index'],
            selected_tier=None if chosen is None else chosen['tier'],selected_iou=None if chosen is None else chosen['iou'],
            lost_count=s['lost_count'],last_observed_bbox=deepcopy(s['last_bbox']),last_observed_frame=s['last_observed_frame'],
            last_high_frame=s['last_high_frame'],consecutive_low_matches=s['consecutive_low_matches'],
            true_gap=None if s['last_observed_frame'] is None else fid-s['last_observed_frame'],reason=reason,candidates=cs,
            input_parents=deepcopy(input_parents),tracker_before_sha256=before,tracker_after_sha256=digest(s))
        digest(decision)  # Reject any non-JSON/nonfinite extra metadata before state mutation.
        self.state=s
        return decision
