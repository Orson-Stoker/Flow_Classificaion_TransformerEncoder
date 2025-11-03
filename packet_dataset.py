from packet_extractor import *
from torch.utils.data import Dataset,DataLoader

class PacketSequenceDataset(Dataset):
    def __init__(self, pcap_files, labels, max_flows=1000):
        self.feature_extractor = FlowFeatureExtractor()
        self.sequences = []
        self.labels = []
        
        
        for pcap_file, label in zip(pcap_files, labels):
            sequence = self.feature_extractor.extract_flow_features(pcap_file,max_flows)
            self.sequences.extend(sequence)
            self.labels.extend([label] * len(sequence))

    def process_pcap_file(self, pcap_file):
        packets = rdpcap(pcap_file)      
        sequence = []
           
        for packet in packets:
            features = self.feature_extractor.extract_packet_features(packet)
            sequence.append(features)

            if len(sequence)>=self.max_sequence_len:
                return sequence
            
        return sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence,label

# pcap_files=[r"data\0.pcap"]
# labels=[0]

# dataset=PacketSequenceDataset(pcap_files,labels,max_flows=10)
# dataloader=DataLoader(dataset,32,shuffle=True)
# for data in dataloader:
#     sequence,label=data
#     print(f"Sequences shape: {sequence.shape}, Labels shape: {label.shape}")