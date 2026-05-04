import tensorflow as tf
from tensorflow.keras import layers

def build_stats_nn(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    return tf.keras.Model(inp, x, name='STATS_NN')

def build_pkt_nn(seq_len, mag_input_dim):
    inp = tf.keras.Input(shape=(seq_len,))
    x = layers.Embedding(input_dim=mag_input_dim, output_dim=32)(inp)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    return tf.keras.Model(inp, x, name='PKT_NN')

def build_pyl_nn(seq_len):
    inp = tf.keras.Input(shape=(seq_len,))
    x = layers.Embedding(input_dim=256, output_dim=32)(inp)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    return tf.keras.Model(inp, x, name='PYL_NN')

def build_aligned_model(stats_dim, pkt_len, pyl_len, mag_input_dim, num_classes):
    stats_nn = build_stats_nn(stats_dim)
    pkt_nn = build_pkt_nn(pkt_len, mag_input_dim)
    pyl_nn = build_pyl_nn(pyl_len)

    def encode(x_stats, x_pkt, x_pyl):
        f1 = stats_nn(x_stats)
        f2 = pkt_nn(x_pkt)
        f3 = pyl_nn(x_pyl)
        return f1, f2, f3

    # 融合部分：Dense → BN → Dropout → Dense（输出 64）
    fusion_dense = tf.keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu')
    ], name="FusionMLP")

    # 构建 classifier 模型：输入 64*3 = 192 维，输出 num_classes 分类
    fusion_input = tf.keras.Input(shape=(64,), name='fused_input')
    output = layers.Dense(num_classes, activation='softmax')(fusion_input)
    classifier = tf.keras.Model(inputs=fusion_input, outputs=output, name='ClassifierModel')

    return encode, fusion_dense, classifier, stats_nn, pkt_nn, pyl_nn

