"""Original P normalize body; only imports/entry wrapper are portable."""
from fractions import Fraction
from planner import require,digest

def normalize(entry, ep, control, previous_digest):
    r, d = control, control['decision']
    i = r['input_identity']
    require(r['schema']=='r11_frame_v1' and r['status']=='frame_committed', 'original control schema/status')
    for expected, a, b, c in ((entry['episode'],r['episode'],i['episode'],d['episode']),
                              (entry['frame'],r['frame'],i['frame'],d['frame_index']),
                              (entry['source'],r['source'],i['source'],d['source_segment'])):
        require(expected==a==b==c, 'control identity conflict')
    require(r['episode_input_sha256']==entry['input_ref']['legacy_content_sha256'], 'episode input parent conflict')
    require(r['previous_frame_receipt']==previous_digest, 'original frame parent chain conflict')
    require(i['width']==ep['stream']['width'] and i['height']==ep['stream']['height'], 'stream size conflict')
    require(i['video_sha256']==ep['video']['sha256'] and i['whole_pts_sha256']==ep['whole_pts']['sha256'], 'source/PTS parent conflict')
    require(d['method']=='single_iou_low_v1', 'unknown tracker version')
    require(d['input_parents']['wholePTS']==i['whole_pts_sha256'], 'decision PTS parent conflict')
    require(all(isinstance(v,str) and len(v)==64 for v in d['input_parents'].values()), 'missing opaque input parent')
    tb = Fraction(i['time_base'])
    require(tb>0 and tb==Fraction(ep['stream']['time_base']), 'time base conflict')
    branches=('initialized_detection','associated_detection','associated_low_detection','held_bbox','full_no_detection','reinitialized_identity_uncertain')
    unknown = d['branch'] not in branches or d['branch']=='full_no_detection'
    restart = d['branch']=='reinitialized_identity_uncertain' or d['reason'] in ('third_loss_high_restart','third_loss_cleared_state')
    version={k:r[k] for k in ('producer','numeric_hash','runtime_sha256','episode_input_sha256')}
    require(all(isinstance(v,str) and len(v) in (40,64) for v in version.values()), 'missing input version')
    candidates=[]
    for c in d['candidates']:
        require(set(c)=={'index','class_id','xyxy','score','tier','valid'}, 'candidate field gap')
        candidates.append({('bbox' if k=='xyxy' else k):v for k,v in c.items()})
    return dict(episode=r['episode'],source=r['source'],frame=r['frame'],pts_num=i['pts'],pts_den=1,
                time_base_num=tb.numerator,time_base_den=tb.denominator,W=i['width'],H=i['height'],
                input_version=digest(version),input_version_values=version,track_segment=d['track_segment'],
                branch=d['branch'],selected_index=d['selected_index'],selected_tier=d['selected_tier'],selected_bbox=d['bbox'],
                lost_count=d['lost_count'],restart=restart,association_state='unknown' if unknown else 'known_control',
                reason=d['reason'],source_discontinuity=d['source_discontinuity'],candidates=candidates,required=True,
                receipt_ref={k:entry[k] for k in ('path','bytes','sha256')},
                parent_ref=dict(previous_frame_receipt=r['previous_frame_receipt'],input_parents=d['input_parents']),
                required_ref=entry['required_source_ref'],source_ref=ep['video'])


FRAME_POINTERS = [
    '/schema','/status','/episode','/frame','/source','/producer','/numeric_hash','/runtime_sha256',
    '/episode_input_sha256','/previous_frame_receipt',
    *['/input_identity/'+k for k in ('episode','source','frame','pts','time_base','width','height','video_sha256','whole_pts_sha256')],
    *['/decision/'+k for k in ('episode','frame_index','source_segment','source_discontinuity','track_segment','method','branch','bbox',
                              'selected_index','selected_tier','lost_count','reason','input_parents')],
    *['/decision/candidates/*/'+k for k in ('index','class_id','xyxy','score','tier','valid')],
]


INPUT_POINTERS = ['/schema','/episode','/required','/sources','/source_map','/video/path','/video/sha256',
                  '/whole_pts/path','/whole_pts/sha256','/whole_pts_verified','/local_file_verified',
                  '/stream/width','/stream/height','/stream/time_base']

