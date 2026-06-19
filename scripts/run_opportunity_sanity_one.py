import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset, opportunity_collate
from torch.utils.data import DataLoader


def main():
    gm = json.load(open('dataset/Opportunity/group_map_official.json'))
    root = 'dataset/Opportunity/extracted/OpportunityUCIDataset/dataset'
    files = ['S1-ADL1.dat']
    ds = OpportunityDataset(root=root, seq_len=32, step=32, group_map=gm, files=files, label_col=None, window_ms=500)
    print('Computed seq_len:', ds.seq_len)
    print('files', [p.name for p in ds.files])
    print('n seqs', len(ds))
    if len(ds) == 0:
        print('no seqs')
        return
    dl = DataLoader(ds, batch_size=2, collate_fn=opportunity_collate)
    batch = next(iter(dl))
    print('batch group count', len(batch['streams']))
    for g,t in batch['streams'].items():
        print(g, '->', t.shape)
    print('mask', batch['mask'].shape)
    print('lengths', batch['lengths'])
    print('labels', batch['labels'])

if __name__ == '__main__':
    main()
