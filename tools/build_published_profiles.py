"""Build reproducible cost fixtures from published channel/rate metadata, never data."""
import csv
import json
import math
from pathlib import Path
from distrixsense.config import METHODS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'opportunity++': 'https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.792065/full',
    'openmarcie': 'https://arxiv.org/html/2603.02390v1',
    'nymeria': 'https://arxiv.org/html/2406.09905v1',
}


def stream(name, family, node, channels, rate, *, image=None, audio=False, note='', assumed=''):
    frames = max(1, round(rate*3))
    shape = [frames, channels] if image is None else [frames, channels, *image]
    width, length = channels, frames
    policy = 'identity'
    if image:
        width = channels*64
        policy = 'spatial_pool_8x8'
    if audio:
        width, length = channels*257, max(1, (frames-512)//256+1)
        policy = 'log_spectrum_fft512_hop256_center_false'
    return dict(name=name, family=family, node=node, rate_hz=rate, raw_shape=shape,
                raw_numeric_bytes=math.prod(shape)*(1 if image and family in ('rgb', 'slam', 'eye_images') else 4),
                raw_storage_convention='uint8 RGB/grayscale images; float32 other decoded values; excludes timestamps, codecs and containers',
                feature_frames=length, feature_channels=width, feature_bytes=length*width*4,
                feature_policy=policy, published_evidence=note, assumptions=assumed,
                corpus_compressed_bytes=None)


def inventories():
    opp = [stream(n, n, n+'_gateway', c, 30,
                  note='Section 2.3: body145 + object60 + ambient37 = 242 attributes.',
                  assumed='30 Hz synchronized sensor fixture; aggregate gateway per subsystem, not one physical sensor.')
           for n, c in [('body_sensors',145), ('object_sensors',60), ('ambient_sensors',37)]]
    opp += [stream('rgb','rgb','camera',3,10,image=(480,640),note='Video: 640x480, 10 fps.'),
            stream('body25','body25','camera',75,10,note='BODY25: 25 keypoints, x/y/confidence from video.',
                   assumed='One person; 10 Hz matching video; excludes OpenPose extraction cost.')]
    datasets = [('opportunity++','opportunity++',opp)]
    for protocol in ('bicycle','printer'):
        s=[]
        counts = {'imu':21,'magnetometer':9,'barometer':3,'temperature':3,'spectrometer':24} if protocol=='bicycle' else {'imu':20,'magnetometer':6,'barometer':2,'temperature':4}
        for n,c in counts.items():
            s.append(stream(n,n,n+'_gateway',c,60 if n=='imu' else 10,
                note=f'Table 2 {protocol}: {c} total channels.',
                assumed='Grouped subsystem gateway. Test rate 60 Hz IMU / 10 Hz other; not a claim about native acquisition rate.'))
        if protocol=='bicycle':
            s.append(stream('thermal','thermal','thermal_gateway',2,10,image=(8,8),
                     note='Table 2: two 8x8 thermal arrays.',assumed='10 Hz test rate; arrays grouped.'))
        # Keep RGB and depth separately so their individual modality costs remain visible.
        camera_groups = [('chest_lidar',1),('room_rgbd',2)] if protocol=='bicycle' else [('rgbd_imu',2),('lidar',2),('room_rgbd',2)]
        for n,count in camera_groups:
            for i in range(count):
                for family,c in [('rgb',3),('depth',1)]:
                    s.append(stream(f'{n}{i}_{family}',family,f'{n}{i}',c,8,image=(224,224),
                        note=f'Table 2: {count} {n} camera(s), four RGB/depth channels each.',
                        assumed='224x224 processing resolution and 8 fps fixture, not native capture dimensions. Depth float32.'))
        if protocol=='printer':
            s.append(stream('camera_imu','imu','camera_imu_gateway',20,60,
                note='Table 2 RGBD-IMU total28 channels.',
                assumed='Interpretation: eight RGBD plus twenty IMU channels; 60 Hz test rate; verify release schema before real training.'))
        for i in range(1 if protocol=='bicycle' else 2):
            s.append(stream(f'audio{i}','audio',f'audio{i}',2,16000,audio=True,
                note='Table 2 stereo audio; benchmark uses 16,000 samples per second.',
                assumed='16 kHz benchmark representation; three-second fixture rather than paper one-second baseline clip.'))
        datasets.append(('openmarcie',f'openmarcie_{protocol}',s))
    s=[]
    for device in ('head','observer','left_wrist','right_wrist'):
        wrist='wrist' in device
        note='Appendix A recording profile: full Aria vs miniAria wrist rates.'
        s.append(stream(device+'_rgb','rgb',device,3,10 if wrist else 30,image=(1408,1408),note=note))
        for camera in ('left','right'):
            s.append(stream(f'{device}_slam_{camera}','slam',device,1,20 if wrist else 30,image=(480,640),note=note))
            s.append(stream(f'{device}_imu_{camera}','imu',device,6,800 if camera=='left' else 1000,note=note,
                            assumed='Six acceleration/angular-velocity channels; timestamps excluded from numeric volume.'))
        s.append(stream(device+'_trajectory','trajectory',device,7,1000,note='Appendix B: MPS 6DoF trajectory at 1 kHz.',
                        assumed='XYZ plus quaternion representation, seven float32 values.'))
        if not wrist:
            s += [stream(device+'_eye_images','eye_images',device,1,10,image=(240,320),note=note,
                         assumed='One published 320x240 ET image stream; no doubling for inferred eye crops.'),
                  stream(device+'_magnetometer','magnetometer',device,3,10,note=note),
                  stream(device+'_barometer','barometer',device,1,50,note=note),
                  stream(device+'_audio','audio',device,7,48000,audio=True,note=note),
                  stream(device+'_gaze','gaze',device,3,10,note='Appendix B: calibrated eye gaze.',
                         assumed='Yaw/pitch/depth at ET 10 Hz; exact output schema and availability must be checked locally.')]
    for n,c in [('body_position',69),('body_orientation',92),('body_velocity',69),('body_acceleration',69),('joint_angles',69),('foot_contacts',4)]:
        s.append(stream(n,n,'xsens',c,240,note='Appendix A: XSens body motion240Hz.',
                        assumed='23 segments x3/x4 fixture; joint-angle69/contact4 are explicit output-schema assumptions, not established release dimensions.'))
    s.append(stream('pointcloud','pointcloud','scene',12,1/3,note='Appendix B: static semi-dense scene point cloud.',
                    assumed='One XYZ point-moments context per window (12 float32 features); raw point count varies and is unknown.'))
    s[-1]['raw_shape']=None
    s[-1]['raw_numeric_bytes']=None
    s[-1]['rate_hz']=None  # Static context has no acquisition frame rate.
    s[-1]['feature_policy']='point_moments_xyz'
    datasets.append(('nymeria','nymeria',s))
    return datasets


def build():
    entries=[]
    inventory=[]
    for dataset,profile,streams in inventories():
        for s in streams:
            inventory.append(dict(dataset=dataset,profile=profile,source=SOURCES[dataset],window_seconds=3,**s))
        groups={'all':[s['name'] for s in streams]}
        for family in sorted({s['family'] for s in streams}):
            groups['only_'+family]=[s['name'] for s in streams if s['family']==family]
        entries.append(dict(dataset=profile,window_seconds=3,stride_seconds=1.5,
            description=f'{dataset} published-dimension synthetic feature fixture; see modality_inventory.json for field-specific assumptions. Random sensor weights. Feature extraction excluded. All derived/observer streams is a cost stress test, not a leakage-safe task protocol.',
            config=dict(modalities={s['name']:s['feature_channels'] for s in streams},classes=12 if dataset=='openmarcie' else 10,
                        hidden=32,heads=4,layers=2,resample_length=32,tokens=8,bank_size=32,dropout=0,modality_dropout=0),
            inputs={s['name']:{'frames':s['feature_frames']} for s in streams},modality_sets=groups,
            topology=dict(stream_devices={s['name']:s['node'] for s in streams},
                edges={s['node']:dict(bandwidth_mbps=10,latency_ms=2,overhead_bytes_per_packet=28) for s in streams})))
    return dict(methods=[m for m in METHODS if m!='imagebind'], seeds=[0], datasets=entries,
        cores=[dict(name=name,checkpoint='../checkpoints/'+folder,dtype='bfloat16',decode_tokens=8,prompt_style='auto')
               for name,folder in [('gemma3_1b','gemma-3-1b-it'),('qwen3_4b','Qwen3-4B'),('qwen3_8b','Qwen3-8B')]]),inventory


def main():
    spec,inventory=build()
    (ROOT/'configs/published_matrix.json').write_text(json.dumps(spec,indent=2)+'\n')
    folder=ROOT/'runs/published-dummy'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'modality_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    with (folder/'modality_inventory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    print(f'{len(inventory)} streams; {sum(len(d["modality_sets"]) for d in spec["datasets"])*len(spec["cores"])*len(spec["methods"])} cost combinations')


if __name__=='__main__':
    main()
