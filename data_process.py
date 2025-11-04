from packet_extractor import *
import os,csv


def save_to_csv(path,pcap_file,sequences,label,feature_names):

    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = os.path.splitext(pcap_file)[0] + ".csv"
    
    with open(os.path.join(path,filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        for sequence in sequences:
            writer.writerow(sequence+[label])
    print(f"{os.path.join(path,filename)} saved.")        

def transform_pcap_to_csv(pcapdir="data",max_flows=100,save_dir="features"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  

    feature_extractor = FlowFeatureExtractor()
    feature_names=feature_extractor.get_feature_names()+["label"]
    
    categories=os.listdir(pcapdir)  

    for category in categories:
        root=os.path.join(pcapdir,category)
        pcap_files=os.listdir(root)

        for pcap_file in pcap_files:
            path=os.path.join(root,pcap_file)
            sequences = feature_extractor.extract_flow_features(path,max_flows)
            save_to_csv(os.path.join(save_dir,category),pcap_file,sequences,category,feature_names)

transform_pcap_to_csv()

