import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset


def pretty(arr, maxrows=5):
    if arr is None:
        return 'None'
    a = np.array(arr)
    if a.size == 0:
        return 'empty'
    if a.ndim == 1:
        return np.array2string(a[:maxrows], precision=4, separator=', ')
    # 2D
    rows = min(a.shape[0], maxrows)
    return '\n' + '\n'.join([np.array2string(r, precision=4, separator=', ') for r in a[:rows]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='S1-ADL1.dat')
    p.add_argument('--index', type=int, default=0, help='window index into dataset')
    p.add_argument('--root', default='dataset/Opportunity/extracted/OpportunityUCIDataset/dataset')
    p.add_argument('--window_ms', type=float, default=500)
    p.add_argument('--label_col', type=int, default=243)
    args = p.parse_args()

    gm_path = 'dataset/Opportunity/group_map_official.json'
    print('Loading group_map from', gm_path)
    gm = json.load(open(gm_path))

    ds = OpportunityDataset(root=args.root, seq_len=32, step=32, group_map=gm, files=[args.file], label_col=args.label_col, window_ms=args.window_ms)
    print('Dataset windows:', len(ds), 'seq_len', ds.seq_len)
    if len(ds) == 0:
        print('no windows')
        return

    print('\nFetching sample index', args.index)
    item = ds[args.index]

    print('\nMetadata:')
    for k in ('file','start','length'):
        print(' ', k, ':', item.get(k))

    print('\nRaw sequence shape:', None if item['sequence'] is None else item['sequence'].shape)
    print('Raw sequence (first rows):', pretty(item['sequence'][:5].numpy() if hasattr(item['sequence'],'numpy') else item['sequence'][:5]))

    print('\nLabels (per-timestep, up to 20):')
    # labels are not returned per timestep when label_col given; dataset stores label in 'label' and per-timestep list only internally
    print(' label (window majority):', item.get('label'))

    print('\nStreams:')
    for g, t in item['streams'].items():
        print(' -', g, 'shape', t.shape)
        print('   first rows:', pretty(t[:5].numpy() if hasattr(t,'numpy') else t[:5]))

    print('\nDone')

if __name__ == '__main__':
    main()
