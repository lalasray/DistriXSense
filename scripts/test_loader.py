import os
import sys
try:
    import torch
except Exception as e:
    print('torch import error:', e)
    sys.exit(1)

from dataloaders.casas_pytorch_dataset import CASASSequenceDataset

root = 'dataset/CASAS/data/data'
print('root exists:', os.path.isdir(root))
ds = CASASSequenceDataset(root, seq_len=128)
print('num_files:', len(ds.files))
print('sensor_vocab:', len(ds.idx2sensor))
print('num_sequences:', len(ds))
if len(ds) > 0:
    item = ds[0]
    print('example file,start,events_shape,labels_len:', item['file'], item['start'], item['events'].shape, len(item['labels']))
print('torch.cuda.is_available:', torch.cuda.is_available())
