import os
import glob
import numpy as np
import binascii
from tqdm import tqdm
from scapy.all import PcapReader


def extract_layers(payload):
    dic = {payload.name: payload}
    payload = payload.payload
    while payload.name != "NoPayload":
        dic[payload.name] = payload
        payload = payload.payload
    return dic


def int_generation(raw_hex, length=64):
    chunks = [raw_hex[i:i+2] for i in range(0, len(raw_hex), 2)]
    ints = [int(c, 16) for c in chunks[:length]]
    ints = np.array(ints, dtype=np.int64)
    return np.pad(ints, (0, length - len(ints)), "constant")


def process_single_pcap(filename, output_dir):
    basename = os.path.basename(filename).split(".")[0]

    time_seq, length_seq = [], []
    ip_flag_seq, tcp_flag_seq, udp_len_seq = [], [], []
    payload_int_seq = []

    with PcapReader(filename) as fdesc:
        while True:
            try:
                pkt = fdesc.read_packet()
                if pkt is None:
                    break

                layers = extract_layers(pkt)
                t = float(pkt.time)

                # TCP
                if "TCP" in layers:
                    tcp_flag = layers["TCP"].flags.value
                    udp_len = 0
                    length = 0 if layers["TCP"].payload.name == "NoPayload" else len(layers["TCP"].payload)
                    raw_hex = binascii.hexlify(bytes(layers["TCP"])).decode()[24:24 + 128 * 2]

                # UDP
                elif "UDP" in layers:
                    tcp_flag = 0
                    udp_len = layers["UDP"].len
                    length = 0 if layers["UDP"].payload.name == "NoPayload" else len(layers["UDP"].payload)
                    raw_hex = binascii.hexlify(bytes(layers["UDP"])).decode()[8:8 + 128 * 2]

                else:
                    continue

                ip_flag = layers["IP"].flags.value

                time_seq.append(t)
                length_seq.append(length)
                ip_flag_seq.append(ip_flag)
                tcp_flag_seq.append(tcp_flag)
                udp_len_seq.append(udp_len)
                payload_int_seq.append(int_generation(raw_hex))

            except EOFError:
                break

    if len(time_seq) < 100:
        return

    # ---------- 时间差 ----------
    time_seq = np.array(time_seq)
    time_delta = np.insert(time_seq[1:] - time_seq[:-1], 0, 0)

    length_seq = np.array(length_seq)
    ip_flag_seq = np.array(ip_flag_seq)
    tcp_flag_seq = np.array(tcp_flag_seq)
    udp_len_seq = np.array(udp_len_seq)
    payload_int_seq = np.array(payload_int_seq)  # [N, 64]

    total_packets = len(length_seq)
    instance_num = total_packets // 100

    if instance_num == 0:
        return

    # 截断
    time_delta = time_delta[: instance_num * 100].reshape(-1, 100, 1)
    length_seq = length_seq[: instance_num * 100].reshape(-1, 100, 1)
    ip_flag_seq = ip_flag_seq[: instance_num * 100].reshape(-1, 100, 1)
    tcp_flag_seq = tcp_flag_seq[: instance_num * 100].reshape(-1, 100, 1)
    udp_len_seq = udp_len_seq[: instance_num * 100].reshape(-1, 100, 1)
    payload_int_seq = payload_int_seq[: instance_num * 100].reshape(-1, 100, 64)

    # 合并特征 → [num_shortflows, 100, 69]
    shortflows = np.concatenate([
        time_delta,
        length_seq,
        ip_flag_seq,
        tcp_flag_seq,
        udp_len_seq,
        payload_int_seq
    ], axis=-1)

    # ----------- 保存一个 npy 文件 -----------
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{basename}.npy")
    np.save(save_path, shortflows)

    print(f"[OK] {filename} → {save_path} (shape = {shortflows.shape})")


# ------------------------------------------------------------
#   主程序
# ------------------------------------------------------------
if __name__ == "__main__":

    data_root = "data"
    output_root = "features"

    subdirs = sorted([d for d in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, d))])

    print("目录映射：")
    for i, d in enumerate(subdirs):
        print(f"{i}: {d}")

    for idx, sub in enumerate(subdirs):
        input_dir = os.path.join(data_root, sub)
        output_dir = os.path.join(output_root, str(idx))

        pcaps = glob.glob(os.path.join(input_dir, "**/*.pcap"), recursive=True)

        print(f"\n{sub} → features/{idx}, 共 {len(pcaps)} 个 PCAP")

        for pcap in tqdm(pcaps, desc=sub):
            process_single_pcap(pcap, output_dir)

    print("\n全部完成！")
