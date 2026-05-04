import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(csv_path, max_pkt_len=20, max_payload_len=50, test_size=0.2, random_state=42):
    print("正在加载数据...")
    df = pd.read_csv(csv_path)

    print("编码标签...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    num_classes = len(np.unique(y))
    print(f"发现 {num_classes} 个类别。")

    # --- PKT特征处理 ---
    print("正在处理 PKT 特征...")

    def parse_pktlen_feature(pktlen_series):
        sequences = []
        for s in pktlen_series:
            try:
                seq = [int(val.strip()) for val in str(s).split(',') if val.strip()]
                sequences.append(seq if seq else [0])
            except:
                sequences.append([0])
        magnitudes = [[abs(x) for x in seq] for seq in sequences]
        directions = [[1 if x >= 0 else 0 for x in seq] for seq in sequences]
        return magnitudes, directions

    pktlen_mag, pktlen_dir = parse_pktlen_feature(df['pktlen'])

    max_vocab_size = 4096
    pktlen_mag_capped = [[min(val, max_vocab_size - 1) for val in seq] for seq in pktlen_mag]
    mag_input_dim = max_vocab_size
    print(f"PKT 幅值词汇表上限设置为: {mag_input_dim}")

    max_pkt_len = min(max(len(s) for s in pktlen_mag_capped) or 1, max_pkt_len)
    print(f"PKT 序列长度上限设置为: {max_pkt_len}")
    X_pkt_mag = pad_sequences(pktlen_mag_capped, maxlen=max_pkt_len, padding='post', truncating='post')
    X_pkt_dir = pad_sequences(pktlen_dir, maxlen=max_pkt_len, padding='post', truncating='post')

    # --- PYL特征处理 ---
    print("正在处理 PYL 特征...")

    def hex_payload_to_byte_sequence(payload, max_len):
        if isinstance(payload, str) and payload != '-' and len(payload) > 0:
            try:
                byte_array = bytes.fromhex(payload[:max_len * 2])
                return list(byte_array)
            except ValueError:
                return [0] * max_len
        else:
            return [0] * max_len

    def process_combined_payload(fwd_payloads, bwd_payloads, max_allowed_len=max_payload_len):
        max_len_in_data = max(
            max(len(bytes.fromhex(p)) if isinstance(p, str) and p != '-' else 0 for p in fwd_payloads),
            max(len(bytes.fromhex(p)) if isinstance(p, str) and p != '-' else 0 for p in bwd_payloads)
        ) or 1

        final_max_len = min(max_len_in_data, max_allowed_len)
        print(f"数据中原始最大 PYL 长度: {max_len_in_data}")
        print(f"将使用截断/填充长度: {final_max_len}")

        combined = []
        for fwd, bwd in zip(fwd_payloads, bwd_payloads):
            fwd_seq = hex_payload_to_byte_sequence(fwd, final_max_len)
            bwd_seq = hex_payload_to_byte_sequence(bwd, final_max_len)
            combined.append(fwd_seq + bwd_seq)

        padded = pad_sequences(combined, maxlen=final_max_len * 2, padding='post', truncating='post')
        return padded, final_max_len * 2

    X_pyl, max_payload_len_combined = process_combined_payload(df["fwd_payload"], df["bwd_payload"], max_payload_len)
    X_pyl = np.expand_dims(X_pyl, axis=-1)

    # --- STATS特征处理 ---
    print("正在处理 STATS 特征...")
    exclude_cols = ['Uuid', 'pktlen', 'fwd_payload', 'bwd_payload', 'category', 'subclass']
    stats_features = [col for col in df.columns if col not in exclude_cols]
    X_stats = df[stats_features].fillna(0).values
    stats_scaler = StandardScaler()
    X_stats = stats_scaler.fit_transform(X_stats)

    # --- 拆分数据 ---
    print("正在拆分数据集...")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y)

    return {
        "X_train_stats": X_stats[train_idx],
        "X_test_stats": X_stats[test_idx],
        "X_train_pkt_mag": X_pkt_mag[train_idx],
        "X_test_pkt_mag": X_pkt_mag[test_idx],
        "X_train_pkt_dir": X_pkt_dir[train_idx],
        "X_test_pkt_dir": X_pkt_dir[test_idx],
        "X_train_pyl": X_pyl[train_idx],
        "X_test_pyl": X_pyl[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "num_classes": num_classes,
        "mag_input_dim": mag_input_dim,
        "max_pkt_len": max_pkt_len,
        "max_payload_len_combined": max_payload_len_combined,
        "label_encoder": label_encoder
    }
