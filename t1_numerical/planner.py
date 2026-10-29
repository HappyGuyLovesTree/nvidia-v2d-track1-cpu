"""S20B-I pure input planner. Standard library only; no models or sample arrays."""
import collections
import math
import statistics
import struct
from fractions import Fraction
from restricted_json import digest

POLICY = {
    'schema': 'r020_fixed_crop_policy_v1', 'ratio_limit': 1.5,
    'crop_expand': 1.2, 'bbox_input_dtype': 'float32_then_float64',
    'input_size': 224, 'normalization': 'original ImageNet RGB',
    'high_definition': 'original valid class_id=0 tier=high; original score>=0.35',
    'low_definition': 'original valid class_id=0 tier=low; original 0.10<score<0.35',
    'unknown_policy': 'isolate; inherit_R0; never bridge',
    'restart_policy': 'new segment; uncertain restart makes entire new segment inherit_R0',
    'threshold_changes_after_prediction': False,
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def integer(x, label, minimum=0):
    require(type(x) is int and x >= minimum, label)
    return x


def box(b):
    require(isinstance(b, list) and len(b) == 4, 'bbox shape')
    require(all(type(x) in (int, float) and math.isfinite(x) for x in b), 'bbox nonfinite/type')
    require(b[2] > b[0] and b[3] > b[1], 'bbox nonpositive')
    return b


def exact_time(row):
    integer(row['pts_num'], 'invalid PTS', minimum=-(2**63))
    integer(row['pts_den'], 'invalid PTS denominator', 1)
    integer(row['time_base_num'], 'invalid time base numerator', 1)
    integer(row['time_base_den'], 'invalid time base denominator', 1)
    return Fraction(row['pts_num'] * row['time_base_num'], row['pts_den'] * row['time_base_den'])


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def transform(union, w, h):
    # Original wrapper casts bbox to float32 before _preprocess converts to float64.
    x1, y1, x2, y2 = map(f32, union)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side0 = max(x2-x1, y2-y1)
    side = side0 * 1.2
    sx, sy = cx - side / 2, cy - side / 2
    extent = [math.floor(sx), math.floor(sy), math.ceil(sx+side), math.ceil(sy+side)]
    a, b, c, d = extent
    return dict(bbox_float32=[x1,y1,x2,y2], square_float=[sx,sy,side], integer_extent=extent,
                padding_ltrb=[max(0,-a),max(0,-b),max(0,c-w),max(0,d-h)],
                source_extent=[max(0,a),max(0,b),min(w,c),min(h,d)],
                cliff=[f32(2*(cx/w)-1),f32(2*(cy/h)-1),f32(side0/max(w,h))],
                wrapper_crop_to_original=[[side/224,0,sx],[0,side/224,sy],[0,0,1]],
                integer_resize_to_original=[[(c-a)/224,0,a],[0,(d-b)/224,b],[0,0,1]])


def row_risks(row):
    risks = []
    if row['association_state'] == 'unknown':
        risks.append('unknown')
    if row['restart']:
        risks.append('restart_identity_uncertain')
    if row['branch'] not in ('initialized_detection', 'associated_detection'):
        risks.append(row['branch'])
    if row['selected_tier'] != 'high':
        risks.append('selected_not_high')
    cs = row['candidates']
    require(isinstance(cs, list), 'missing candidates')
    seen, high, low = set(), [], []
    for c in cs:
        integer(c['index'], 'candidate index')
        require(c['index'] not in seen, 'duplicate candidate index')
        seen.add(c['index'])
        require(type(c['valid']) is bool, 'missing candidate validity')
        if not c['valid']:
            require(c['tier'] == 'rejected', 'invalid candidate tier conflict')
            continue
        require(type(c['class_id']) is int, 'missing person class')
        require(type(c['score']) in (int,float) and math.isfinite(c['score']) and 0 <= c['score'] <= 1, 'invalid score')
        expected = 'high' if c['score'] >= .35 else 'low' if c['score'] > .10 else 'excluded'
        require(c['tier'] == expected, 'original tier/score conflict')
        box(c['bbox'])
        if c['class_id'] == 0 and c['tier'] == 'high':
            high.append(c)
        if c['class_id'] == 0 and c['tier'] == 'low':
            low.append(c)
    if len(high) != 1:
        risks.append('multiple_high_person' if len(high) > 1 else 'no_high_person')
    chosen = next((c for c in cs if c['index'] == row['selected_index']), None)
    if row['selected_index'] is not None:
        require(chosen is not None, 'selected index missing')
        require(chosen['bbox'] == row['selected_bbox'] and chosen['tier'] == row['selected_tier'], 'selected candidate conflict')
        if not chosen['valid'] or chosen['class_id'] != 0:
            risks.append('selected_invalid_person')
    elif row['selected_tier'] is not None:
        raise ValueError('null selection tier conflict')
    if row['lost_count'] != 0:
        risks.append('lost_count_nonzero')
    return risks, len(low)


def plan(rows):
    require(rows, 'empty control sequence')
    groups, previous = [], None
    seen = set()
    for row in rows:
        integer(row['episode'], 'episode')
        integer(row['frame'], 'frame')
        integer(row['W'], 'width', 1)
        integer(row['H'], 'height', 1)
        integer(row['lost_count'], 'lost_count')
        integer(row['track_segment'], 'track segment', -1)
        require(type(row['restart']) is bool, 'restart missing')
        require(row['association_state'] in ('known_control', 'unknown'), 'association state missing')
        require(row.get('receipt_ref') and row.get('parent_ref') and row.get('required_ref') and row.get('source_ref'), 'missing ref')
        box(row['selected_bbox'])
        current_time = exact_time(row)
        identity = (row['episode'], row['source'], row['track_segment'], row['input_version'], row['W'], row['H'])
        require((row['episode'],row['frame']) not in seen, 'duplicate frame')
        seen.add((row['episode'],row['frame']))
        boundary, extra_risks = [], []
        if previous is None:
            boundary.append('sequence_start')
        else:
            prev_identity = (previous['episode'], previous['source'], previous['track_segment'], previous['input_version'], previous['W'], previous['H'])
            labels = ('episode_change','source_change','track_segment_change','input_version_change','width_change','height_change')
            boundary.extend(label for a,b,label in zip(prev_identity,identity,labels) if a != b)
            if row['episode'] == previous['episode']:
                require(row['frame'] > previous['frame'], 'control sequence out of order')
                if row['frame'] != previous['frame'] + 1:
                    boundary.append('frame_gap')
                if current_time <= exact_time(previous):
                    boundary.append('pts_nonmonotonic')
                    extra_risks.append('pts_nonmonotonic')
            if previous['association_state'] == 'unknown':
                boundary.append('after_unknown')
        if row['association_state'] == 'unknown':
            boundary.append('explicit_unknown')
        if row['restart']:
            boundary.append('third_loss_restart')
        if boundary:
            groups.append(dict(key=list(identity), boundary_before=boundary, rows=[], extra_risks=[]))
        groups[-1]['rows'].append(row)
        groups[-1]['extra_risks'].extend(extra_risks)
        previous = row
    segments = []
    for ordinal, group in enumerate(groups):
        rr = group['rows']
        risks = list(group['extra_risks'])
        lows = 0
        for row in rr:
            gates, low = row_risks(row)
            risks.extend(gates)
            lows += low
        union = [min(x['selected_bbox'][0] for x in rr),min(x['selected_bbox'][1] for x in rr),
                 max(x['selected_bbox'][2] for x in rr),max(x['selected_bbox'][3] for x in rr)]
        median = statistics.median(max(x['selected_bbox'][2]-x['selected_bbox'][0],x['selected_bbox'][3]-x['selected_bbox'][1]) for x in rr)
        ratio = max(union[2]-union[0], union[3]-union[1]) / median
        if ratio > POLICY['ratio_limit']:
            risks.append('union_ratio_exceeded')
        risks = list(dict.fromkeys(risks))
        segment_id = digest(dict(key=group['key'], ordinal=ordinal, first_ref=rr[0]['receipt_ref']))
        action = 'inherit_R0' if risks else 'update'
        segments.append(dict(segment_id=segment_id, key=group['key'], boundary_before=group['boundary_before'],
                             frames=[r['frame'] for r in rr], required_frames=[r['frame'] for r in rr if r['required']],
                             first_frame=rr[0]['frame'], last_frame=rr[-1]['frame'],
                             first_time=[rr[0][k] for k in ('pts_num','pts_den','time_base_num','time_base_den')],
                             last_time=[rr[-1][k] for k in ('pts_num','pts_den','time_base_num','time_base_den')],
                             union=union, median_original_side=median, union_ratio=ratio,
                             transform=transform(union, rr[0]['W'], rr[0]['H']),
                             action=action, first_reason=risks[0] if risks else None, risks=risks,
                             low_person_candidate_occurrences=lows,
                             source_refs=[r['receipt_ref'] for r in rr],
                             per_frame=[dict(episode=r['episode'],source=r['source'],frame=r['frame'],action=action,
                                             pts_num=r['pts_num'],pts_den=r['pts_den'],
                                             time_base_num=r['time_base_num'],time_base_den=r['time_base_den'],
                                             W=r['W'],H=r['H'],source_ref=r['source_ref'],
                                             input_version=r['input_version'],receipt_ref=r['receipt_ref']) for r in rr]))
    return dict(schema='r020_fixed_crop_plan_v1', policy=POLICY, policy_sha256=digest(POLICY), segments=segments,
                control_count=len(rows), required_count=sum(r['required'] for r in rows),
                update_N=sum(len(s['required_frames']) for s in segments if s['action']=='update'),
                source_count=len({(r['episode'],r['source']) for r in rows}),
                fallback_reason_segments=dict(collections.Counter(s['first_reason'] for s in segments if s['action']!='update')))
