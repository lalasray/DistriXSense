import unittest
from tools.build_published_profiles import build


class PublishedProfilesTests(unittest.TestCase):
    def test_sensor_partition_and_media_shapes(self):
        spec, rows = build()
        opp = [r for r in rows if r['dataset']=='opportunity++']
        self.assertEqual(sum(r['feature_channels'] for r in opp if r['name'].endswith('sensors')),242)
        video = next(r for r in opp if r['name']=='rgb')
        self.assertEqual(video['raw_shape'],[30,3,480,640])
        self.assertEqual(video['raw_numeric_bytes'],27648000)
        self.assertEqual(video['feature_bytes'],30*192*4)
        audio = next(r for r in rows if r['name']=='head_audio')
        self.assertEqual(audio['raw_shape'],[144000,7])
        self.assertEqual(audio['feature_channels'],1799)
        self.assertEqual(audio['feature_frames'],561)
        wrists = [r for r in rows if r['dataset']=='nymeria' and 'wrist' in r['node']]
        self.assertFalse(any(r['family'] in ('audio','barometer','magnetometer','eye_images') for r in wrists))
        points = next(r for r in rows if r['name']=='pointcloud')
        self.assertIsNone(points['raw_numeric_bytes'])
        self.assertTrue(all(r['corpus_compressed_bytes'] is None for r in rows))

    def test_each_stream_covered_once_per_family_without_cross_protocol_overlap(self):
        spec, rows = build()
        for entry in spec['datasets']:
            streams = entry['config']['modalities']
            family_streams = [n for group,names in entry['modality_sets'].items() if group!='all' for n in names]
            self.assertEqual(sorted(family_streams),sorted(streams))
            self.assertEqual(set(streams),set(entry['inputs']))
            self.assertEqual(set(streams),set(entry['topology']['stream_devices']))
        self.assertEqual(len(spec['cores']),3)
        self.assertEqual(len(spec['methods']),21)
