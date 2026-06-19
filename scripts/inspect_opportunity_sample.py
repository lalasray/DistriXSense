import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset


def describe_tensor(t):
    if t is None:
        return 'None'
    if hasattr(t, 'numpy'):
        a = t.numpy()
    else:
        a = np.array(t)
    shape = a.shape
    dtype = str(a.dtype)
    total = a.size
    try:
        n_nan = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    except Exception:
        n_nan = 0
    # compute min/max ignoring NaN for numeric
    try:
        if np.issubdtype(a.dtype, np.number):
            valid = a[~np.isnan(a)] if a.size and np.issubdtype(a.dtype, np.floating) else a
            if valid.size:
                vmin = float(np.min(valid))
                vmax = float(np.max(valid))
            else:
                vmin = vmax = None
        else:
            vmin = vmax = None
    except Exception:
        vmin = vmax = None
    return dict(shape=shape, dtype=dtype, total=total, n_nan=n_nan, min=vmin, max=vmax)


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

    seq = item.get('sequence')
    seq_desc = describe_tensor(seq)
    print('\nSequence:')
    print(' ', 'shape:', seq_desc['shape'], 'dtype:', seq_desc['dtype'], 'total:', seq_desc['total'], 'n_nan:', seq_desc['n_nan'])

    print('\nLabel (window majority):', item.get('label'))

    print('\nStreams summary:')
    for g, t in item['streams'].items():
        desc = describe_tensor(t)
        print(f" - {g}: shape={desc['shape']}, dtype={desc['dtype']}, total={desc['total']}, n_nan={desc['n_nan']}, min={desc['min']}, max={desc['max']}")

    print('\nDone')

if __name__ == '__main__':
    main()
