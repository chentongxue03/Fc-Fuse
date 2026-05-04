import tensorflow as tf

def classification_loss(y_true, y_pred):
    """标准交叉熵分类损失"""
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return loss


def intra_modal_distance_loss(features):
    """组内模态对齐损失（DN1/DN2）"""
    f1, f2, f3 = features
    # 使用余弦距离代替欧氏距离，更稳定
    d12 = 1 - tf.reduce_mean(tf.keras.losses.cosine_similarity(f1, f2))
    d13 = 1 - tf.reduce_mean(tf.keras.losses.cosine_similarity(f1, f3))
    d23 = 1 - tf.reduce_mean(tf.keras.losses.cosine_similarity(f2, f3))
    return (d12 + d13 + d23) / 3.0


def inter_sample_repulsion_loss(features1, features2, margin=1.0):
    """组间模态排斥损失（DJ）"""
    f1_1, f2_1, f3_1 = features1
    f1_2, f2_2, f3_2 = features2

    # 计算相同模态特征的相似度
    sim_f1 = tf.reduce_mean(tf.keras.losses.cosine_similarity(f1_1, f1_2))
    sim_f2 = tf.reduce_mean(tf.keras.losses.cosine_similarity(f2_1, f2_2))
    sim_f3 = tf.reduce_mean(tf.keras.losses.cosine_similarity(f3_1, f3_2))

    # 使用margin确保距离足够大
    repulsion_loss = tf.maximum(0.0, margin - (sim_f1 + sim_f2 + sim_f3) / 3.0)
    return repulsion_loss


def cross_modal_alignment_loss(features1, features2, pred1, pred2, y_true1, y_true2,
                               margin=1.0, alpha=1.0, beta=1.0):
    """完整的跨模态对齐损失函数"""
    # 分类损失
    CL1 = tf.reduce_mean(classification_loss(y_true1, pred1))
    CL2 = tf.reduce_mean(classification_loss(y_true2, pred2))

    # 组内模态对齐损失
    DN1 = intra_modal_distance_loss(features1)
    DN2 = intra_modal_distance_loss(features2)

    # 组间模态排斥损失
    DJ = inter_sample_repulsion_loss(features1, features2, margin)

    # 总损失 = 分类损失 + 组内对齐损失 + 组间排斥损失
    total_loss = CL1 + CL2 + alpha * (DN1 + DN2) + beta * DJ

    return total_loss, {
        "CL1": CL1, "CL2": CL2,
        "DN1": DN1, "DN2": DN2,
        "DJ": DJ, "Total": total_loss
    }

