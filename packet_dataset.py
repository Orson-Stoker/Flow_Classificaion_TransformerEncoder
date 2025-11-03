from packet_extractor import *
from torch.utils.data import Dataset,DataLoader,Subset
import os

class PacketSequenceDataset(Dataset):
    def __init__(self, pcapdir, max_flows=2):
        self.feature_extractor = FlowFeatureExtractor()
        self.sequences = []
        self.labels = []

        categories=os.listdir(pcapdir)  

        for category in categories:
            root=os.path.join(pcapdir,category)
            pcap_files=os.listdir(root)

            for pcap_file in pcap_files:
                path=os.path.join(root,pcap_file)
                sequence = self.feature_extractor.extract_flow_features(path,max_flows)
                self.sequences.extend(sequence)
                self.labels.extend([int(category)] * len(sequence))

    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence,label

