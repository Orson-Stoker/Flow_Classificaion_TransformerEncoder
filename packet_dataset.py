from torch.utils.data import Dataset,DataLoader,Subset
import os
import numpy as np
import torch

class PacketSequenceDataset(Dataset):
    def __init__(self,npy_dir):
        self.sequences = []
        self.labels = []
        categories=os.listdir(npy_dir)  

        for category in categories:
            root=os.path.join(npy_dir,category)
            npy_files=os.listdir(root)

            for npy_file in npy_files:
                path=os.path.join(root,npy_file)
                sequence=np.load(path)
                print(f"Loaded {len(sequence)} samples from {path}...")
                self.sequences.extend(sequence)
                self.labels.extend([int(category)] * len(sequence))

    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence,label


