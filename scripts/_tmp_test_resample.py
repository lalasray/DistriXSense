import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.opportunity_pytorch_dataset import OpportunityDataset

print('Creating dataset with resample_on_drift=True')
ds = OpportunityDataset(root='dataset/Opportunity/extracted/OpportunityUCIDataset/dataset', seq_len=15, resample_on_drift=True)
print('Seqs:', len(ds))
item = ds[0]
print('sequence shape', item['sequence'].shape)
print('file', item['file'], 'start', item['start'], 'label', item['label'])
