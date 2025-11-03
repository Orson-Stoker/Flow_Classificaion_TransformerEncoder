import torch
import numpy as np
import statistics
from scapy.all import *
from collections import Counter


class FlowFeatureExtractor:
    def __init__(self):
        self.protocol_map = {
            'TCP': 0, 'UDP': 1, 'ICMP': 2, 'HTTP': 3, 
            'HTTPS': 4, 'DNS': 5, 'ARP': 6, 'OTHER': 7
        }

        self.flow_cache = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'end_time': None,
            'src_ip': None,
            'dst_ip': None,
            'protocol': None
        })


    def extract_packet_features(self,packet):
        features=[]
        if IP in packet:
            ip_layer = packet[IP]
            protocol = self.get_protocol_name(packet)
            features.append(self.protocol_map.get(protocol, 7)) 
            features.append(ip_layer.ttl)                      
            features.append(len(packet))                       
            features.append(ip_layer.tos)                       
        else:
            features.extend([7, 0, 0, 0])  

        
        if TCP in packet:
            tcp = packet[TCP]
            features.extend([
                tcp.sport,           
                tcp.dport,           
                tcp.window,          
                tcp.dataofs << 2 if tcp.dataofs else 0,  
            ])
      
            tcp_flags = self.extract_tcp_flags(tcp)
            features.extend(tcp_flags)
            
        elif UDP in packet:
            udp = packet[UDP]
            features.extend([
                udp.sport,
                udp.dport,
                0, 0  
            ])
            features.extend([0] * 6)  
        else:
            features.extend([0] * 10)  


        return features


    def get_protocol_name(self, packet):

        if TCP in packet:
            if packet[TCP].dport == 80 or packet[TCP].sport == 80:
                return 'HTTP'
            elif packet[TCP].dport == 443 or packet[TCP].sport == 443:
                return 'HTTPS'
            else:
                return 'TCP'
        elif UDP in packet:
            if packet[UDP].dport == 53 or packet[UDP].sport == 53:
                return 'DNS'
            else:
                return 'UDP'
        elif ICMP in packet:
            return 'ICMP'
        elif ARP in packet:
            return 'ARP'
        else:
            return 'OTHER' 


    def extract_tcp_flags(self, tcp_layer):

        flags = tcp_layer.flags
        return [
            int(flags & 0x02 > 0),  # SYN
            int(flags & 0x10 > 0),  # ACK
            int(flags & 0x01 > 0),  # FIN
            int(flags & 0x04 > 0),  # RST
            int(flags & 0x08 > 0),  # PSH
            int(flags & 0x20 > 0),  # URG
        ]
   
    def extract_flow_features(self, pcap_file, max_flows=1000):
        print(f"Processing {pcap_file}...")
        packets = rdpcap(pcap_file)
        
        self.group_packets_to_flows(packets)
        
   
        flow_features = []
        flow_labels = [] 
        
        for flow_key, flow_data in self.flow_cache.items():
            if len(flow_data['packets']) < 2: 
                continue
                
            features = self.compute_flow_statistics(flow_data)
            if features:
                flow_features.append(features)
                
            if len(flow_features) >= max_flows:
                break
                
        return flow_features

    def group_packets_to_flows(self, packets):

        for packet in packets:
            if IP not in packet:
                continue
                
            flow_key = self.get_flow_key(packet)
            flow_data = self.flow_cache[flow_key]
            
            if flow_data['start_time'] is None:
                flow_data.update({
                    'start_time': packet.time,
                    'src_ip': packet[IP].src,
                    'dst_ip': packet[IP].dst,
                    'protocol': self.get_protocol_name(packet)
                })
            
            flow_data['packets'].append(packet)
            flow_data['end_time'] = packet.time
    
    def get_flow_key(self, packet):

        if IP in packet:
            proto = packet[IP].proto
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            if TCP in packet:
                return (src_ip, dst_ip, proto, packet[TCP].sport, packet[TCP].dport, 'TCP')
            elif UDP in packet:
                return (src_ip, dst_ip, proto, packet[UDP].sport, packet[UDP].dport, 'UDP')
            else:
                return (src_ip, dst_ip, proto, 0, 0, 'OTHER')
        return None
    
    def compute_flow_statistics(self, flow_data):

        packets = flow_data['packets']
        if len(packets) < 2:
            return None
            
        
        packets.sort(key=lambda x: x.time)
        
        src_ip = flow_data['src_ip']
        forward_pkts = [p for p in packets if p[IP].src == src_ip]
        backward_pkts = [p for p in packets if p[IP].dst == src_ip]
        
        features = []

        
        duration = flow_data['end_time'] - flow_data['start_time']
        features.extend([
            duration,                           
            len(packets),                       
            len(forward_pkts),                  
            len(backward_pkts),                 
        ])

        
        pkt_lengths = [len(p) for p in packets]
        fwd_lengths = [len(p) for p in forward_pkts]
        bwd_lengths = [len(p) for p in backward_pkts]
        
        features.extend(self.compute_statistics(pkt_lengths, 'total'))
        features.extend(self.compute_statistics(fwd_lengths, 'fwd'))
        features.extend(self.compute_statistics(bwd_lengths, 'bwd'))

        
        iat = [packets[i].time - packets[i-1].time for i in range(1, len(packets))]
        fwd_iat = [forward_pkts[i].time - forward_pkts[i-1].time 
                   for i in range(1, len(forward_pkts))] if len(forward_pkts) > 1 else []
        bwd_iat = [backward_pkts[i].time - backward_pkts[i-1].time 
                   for i in range(1, len(backward_pkts))] if len(backward_pkts) > 1 else []
        
        features.extend(self.compute_statistics(iat, 'iat'))
        features.extend(self.compute_statistics(fwd_iat, 'fwd_iat'))
        features.extend(self.compute_statistics(bwd_iat, 'bwd_iat'))

        
        tcp_features = self.extract_tcp_flow_features(packets, src_ip)
        features.extend(tcp_features)

    
        protocol_features = self.extract_protocol_specific_features(flow_data)
        features.extend(protocol_features)

        
        behavior_features = self.extract_behavior_features(packets, src_ip)
        features.extend(behavior_features)

        return features

    def compute_statistics(self, data, prefix):
        """计算数据的统计特征"""
        if not data:
            return [0, 0, 0, 0, 0]  # sum, mean, std, max, min
        
        return [
            sum(data),
            statistics.mean(data),
            statistics.stdev(data) if len(data) > 1 else 0,
            max(data),
            min(data)
        ]

    def extract_tcp_flow_features(self, packets, src_ip):
    
        syn_count = fin_count = rst_count = psh_count = ack_count = urg_count = 0
        fwd_syn = fwd_fin = fwd_rst = fwd_psh = fwd_ack = fwd_urg = 0
        bwd_syn = bwd_fin = bwd_rst = bwd_psh = bwd_ack = bwd_urg = 0
        
        for pkt in packets:
            if TCP in pkt:
                tcp = pkt[TCP]
                flags = tcp.flags
                is_forward = pkt[IP].src == src_ip
                
                
                syn_count += 1 if flags & 0x02 else 0
                fin_count += 1 if flags & 0x01 else 0
                rst_count += 1 if flags & 0x04 else 0
                psh_count += 1 if flags & 0x08 else 0
                ack_count += 1 if flags & 0x10 else 0
                urg_count += 1 if flags & 0x20 else 0
                
                
                if is_forward:
                    fwd_syn += 1 if flags & 0x02 else 0
                    fwd_fin += 1 if flags & 0x01 else 0
                    fwd_rst += 1 if flags & 0x04 else 0
                    fwd_psh += 1 if flags & 0x08 else 0
                    fwd_ack += 1 if flags & 0x10 else 0
                    fwd_urg += 1 if flags & 0x20 else 0
                else:
                    bwd_syn += 1 if flags & 0x02 else 0
                    bwd_fin += 1 if flags & 0x01 else 0
                    bwd_rst += 1 if flags & 0x04 else 0
                    bwd_psh += 1 if flags & 0x08 else 0
                    bwd_ack += 1 if flags & 0x10 else 0
                    bwd_urg += 1 if flags & 0x20 else 0
        
        return [
            syn_count, fin_count, rst_count, psh_count, ack_count, urg_count,
            fwd_syn, fwd_fin, fwd_rst, fwd_psh, fwd_ack, fwd_urg,
            bwd_syn, bwd_fin, bwd_rst, bwd_psh, bwd_ack, bwd_urg
        ]

    def extract_protocol_specific_features(self, flow_data):
    
        packets = flow_data['packets']
        protocol = flow_data['protocol']
        features = []
        
        
        if TCP in packets[0] or UDP in packets[0]:
            ports = []
            for pkt in packets:
                if TCP in pkt:
                    ports.extend([pkt[TCP].sport, pkt[TCP].dport])
                elif UDP in pkt:
                    ports.extend([pkt[UDP].sport, pkt[UDP].dport])
            
            if ports:
                features.extend([
                    min(ports), max(ports), statistics.mean(ports),
                    len(set(ports)) 
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
            
        return features

    def extract_behavior_features(self, packets, src_ip):
      
        sizes = [len(p) for p in packets]
        size_bins = [0, 64, 128, 256, 512, 1024, 1500]
        size_hist = np.histogram(sizes, bins=size_bins)[0].tolist()
        
     
        duration = packets[-1].time - packets[0].time
        packet_rate = len(packets) / duration if duration > 0 else 0
        byte_rate = sum(sizes) / duration if duration > 0 else 0
        
      
        active_duration = sum([packets[i].time - packets[i-1].time 
                             for i in range(1, len(packets))])
        active_ratio = active_duration / duration if duration > 0 else 0
        
        features = size_hist + [packet_rate, byte_rate, active_ratio]
        return features

    def get_feature_names(self):
        
        base_names = [
            'duration', 'total_packets', 'fwd_packets', 'bwd_packets'
        ]
        
        stat_types = ['total_len', 'fwd_len', 'bwd_len', 'iat', 'fwd_iat', 'bwd_iat']
        stat_metrics = ['sum', 'mean', 'std', 'max', 'min']
        
        for stat in stat_types:
            for metric in stat_metrics:
                base_names.append(f'{stat}_{metric}')
        
        tcp_names = [
            'syn', 'fin', 'rst', 'psh', 'ack', 'urg',
            'fwd_syn', 'fwd_fin', 'fwd_rst', 'fwd_psh', 'fwd_ack', 'fwd_urg',
            'bwd_syn', 'bwd_fin', 'bwd_rst', 'bwd_psh', 'bwd_ack', 'bwd_urg'
        ]
        
        protocol_names = ['min_port', 'max_port', 'mean_port', 'unique_ports']
        
        behavior_names = [
            'size_0_64', 'size_64_128', 'size_128_256', 'size_256_512', 
            'size_512_1024', 'size_1024_1500', 'packet_rate', 'byte_rate', 'active_ratio'
        ]
        
        return base_names + tcp_names + protocol_names + behavior_names


if __name__ == "__main__":
    extractor = FlowFeatureExtractor()
    
    
    flow_features = extractor.extract_flow_features(r"data/0.pcap")
    
    # print(f"提取了 {len(flow_features)} 个流")
    # print(f"每个流特征维度: {len(flow_features[0])}")
    # print(f"特征名称数量: {len(extractor.get_feature_names())}")
    
    # 转换为Transformer输入格式
    # flow_features shape: (num_flows, feature_dim)
    # 需要添加序列维度: (num_flows, 1, feature_dim)
    # print(len(flow_features))
    
